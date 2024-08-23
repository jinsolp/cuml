#=============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

import h5py
import argparse
import numpy as np
import rmm
import os
import pickle as pkl
import scanpy

from cuml.manifold.umap import UMAP
from cuml.cluster import HDBSCAN
from cuml.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

import sklearn

from sklearn import datasets

import cuml

def memmap_bin_file(
    bin_file, dtype, shape=None, mode="r", size_dtype=np.uint32
):
    extent_itemsize = np.dtype(size_dtype).itemsize
    offset = int(extent_itemsize) * 2
    if bin_file is None:
        return None
    if dtype is None:
        dtype = np.float32

    if mode[0] == "r":
        a = np.memmap(bin_file, mode=mode, dtype=size_dtype, shape=(2,))
        if shape is None:
            shape = (a[0], a[1])
        else:
            shape = tuple(
                [
                    aval if sval is None else sval
                    for aval, sval in zip(a, shape)
                ]
            )

        return np.memmap(
            bin_file, mode=mode, dtype=dtype, offset=offset, shape=shape
        )
    elif mode[0] == "w":
        if shape is None:
            raise ValueError("Need to specify shape to map file in write mode")

        print("creating file", bin_file)
        dirname = os.path.dirname(bin_file)
        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True)
        a = np.memmap(bin_file, mode=mode, dtype=size_dtype, shape=(2,))
        a[0] = shape[0]
        a[1] = shape[1]
        a.flush()
        del a
        fp = np.memmap(
            bin_file, mode="r+", dtype=dtype, offset=offset, shape=shape
        )
        return fp



parser = argparse.ArgumentParser()

parser.add_argument("--umap", action="store_true")
parser.add_argument("--hdbscan", action="store_true") 

parser.add_argument("--nnd", action="store_true")
parser.add_argument("--bfk", action="store_true")
parser.add_argument("--auto", action="store_true")

parser.add_argument("--data", default="mnist")

parser.add_argument("--do-batch", action="store_true")
parser.add_argument("--host", action="store_true")
parser.add_argument("--cluster", default=1, type=int)

parser.add_argument("--rows", default=0, type=int)
parser.add_argument("--cols", default=0, type=int)

parser.add_argument("--gd", default=64, type=int)
parser.add_argument("--igd", default=128, type=int)
parser.add_argument("--iters", default=20, type=int)

args = parser.parse_args()

name_to_path = {"gist": "data/gist-960-euclidean.hdf5",
                "sift": "data/sift-128-euclidean.hdf5",
                "mnist": "data/mnist-784-euclidean.hdf5",
                "food": "data/amazon-food.pkl",
                "book": "data/amazon-book.pkl",
                "wiki": "data/base.88M.fbin",
                "deep": "data/deep-image-96-angular.hdf5",
                "clothes": "data/amazon-clothes.pkl",
                "elec": "data/amazon-elec.pkl",
                "cell": "data/1M_brain_cells_10X.sparse.h5ad",
                "lung": "data/krasnow_hlca_10x.sparse.h5ad"}

if name_to_path[args.data][-4:] == "hdf5":
    hf = h5py.File(name_to_path[args.data], 'r')
    data = np.array(hf['train'])
elif name_to_path[args.data][-3:] == "pkl":
    with open(name_to_path[args.data], 'rb') as file:
        data = pkl.load(file)
elif name_to_path[args.data][-3:] == "bin":
    data = memmap_bin_file(name_to_path[args.data], None)
    data = np.asarray(data)
elif name_to_path[args.data][-4:] == "h5ad":
    adata = scanpy.read(name_to_path[args.data])
    data = adata.X.toarray()
    
# pool = rmm.mr.PoolMemoryResource(
#     rmm.mr.CudaMemoryResource(),
#     initial_pool_size=2**30,
#     maximum_pool_size=2**50
# )
# rmm.mr.set_current_device_resource(pool)


if args.rows != 0:
    data = data[:args.rows, :]
if args.cols != 0:
    data = data[:, :args.cols]

print(args)
print(data.shape, type(data))
rmm.statistics.enable_statistics()

if args.umap:
    if args.nnd:
        umap_nnd = UMAP(n_neighbors=16, build_algo="nn_descent", build_kwds={'nnd_graph_degree': 32, 'nnd_intermediate_graph_degree': 64,
        'nnd_max_iterations': 10, 'nnd_return_distances': True, "nnd_n_clusters": args.cluster}, verbose=5)
        with rmm.statistics.profiler(name="umap nnd"):
            embedding = umap_nnd.fit_transform(data,  data_on_host=args.host)
    elif args.bfk:
        umap_bfk = UMAP(n_neighbors=16, build_algo="brute_force_knn")
        with rmm.statistics.profiler(name="umap bfk"):
            embedding = umap_bfk.fit_transform(data,  data_on_host=args.host)
    elif args.auto:
        umap_nnd = UMAP(n_neighbors=16, build_kwds={'nnd_graph_degree': 32, 'nnd_intermediate_graph_degree': 64,
        'nnd_max_iterations': 10, 'nnd_return_distances': True,  "nnd_n_clusters": args.cluster}, verbose=5)
        with rmm.statistics.profiler(name="umap auto"):
            embedding = umap_nnd.fit_transform(data,  data_on_host=args.host)
    
    print(rmm.statistics.default_profiler_records.report())
    if data.shape[0] > 1000000:
        random_integers = np.random.randint(0, data.shape[0], size=1000000)
        score = cuml.metrics.trustworthiness(data[random_integers, :], embedding[random_integers, :])
    else:
        score = cuml.metrics.trustworthiness(data, embedding)
    print(score)
    

if args.hdbscan:
    hdbscan_nnd = HDBSCAN(min_samples=16, build_algo="nn_descent", build_kwds={'nnd_graph_degree': args.gd, 'nnd_intermediate_graph_degree': args.igd,
        'nnd_max_iterations': args.iters, 'nnd_return_distances': True, "nnd_n_clusters": args.cluster})
    
    with rmm.statistics.profiler(name="hdbscan nnd"):
        # labels_nnd = hdbscan_nnd.fit(data, data_on_host=args.host).labels_
        labels_nnd = hdbscan_nnd.fit(data).labels_
        
    print("Done running nnd\n\n")
    
    if os.path.exists(f"hdbscan_bfk_{args.data}_label.pkl"):
        with open(f"hdbscan_bfk_{args.data}_label.pkl", 'rb') as file:
            labels_bfk = pkl.load(file)
    else:
    
        hdbscan_bfk = HDBSCAN(min_samples=16, build_algo="brute_force_knn")
        with rmm.statistics.profiler(name="hdbscan bfk"):
            labels_bfk = hdbscan_bfk.fit(data).labels_
        
        with open(f"hdbscan_bfk_{args.data}_label.pkl", 'wb') as f:
            pkl.dump(labels_bfk, f)
    
    print(rmm.statistics.default_profiler_records.report())
    
    if data.shape[0] > 1000000:
        random_integers = np.random.randint(0, data.shape[0], size=1000000)
        score = sklearn.metrics.adjusted_rand_score(labels_nnd[random_integers], labels_bfk[random_integers])
        print(score)
        score = sklearn.metrics.adjusted_rand_score(labels_nnd, labels_bfk)
        print(score)
    else:
        score = sklearn.metrics.adjusted_rand_score(labels_nnd, labels_bfk)
        print(score)
    score = adjusted_rand_score(labels_nnd, labels_bfk)
    print(score)
    