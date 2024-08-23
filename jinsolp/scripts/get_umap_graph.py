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

import umap
from cuml.manifold.umap import UMAP as cuUMAP
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import time
from sentence_transformers import SentenceTransformer
import pickle
from sklearn import datasets

# model = SentenceTransformer('all-MiniLM-L6-v2')
# dataset = load_dataset("dstefa/New_York_Times_Topics")
# sentences = []
# labels = []
# for row in dataset['train']:
#     sentences.append(row['text'])
#     labels.append(int(row['topic_id']))

# embeddings = model.encode(sentences, show_progress_bar=True)
# print(embeddings.shape)

with open('data/nytimes-embedding.pkl', 'rb') as file:
    data = pickle.load(file)
with open('data/nytimes-labels.pkl', 'rb') as file:
    labels = pickle.load(file)    
# with open('data/nytimes-embedding.pkl', 'wb') as f:
#     pickle.dump(embeddings, f)
# with open('data/nytimes-labels.pkl', 'wb') as f:
#     pickle.dump(np.array(labels), f)
# quit()
# train_dataset = dataset['train']
# train_images = train_dataset['image']  # This will give you the images
# train_labels = train_dataset['label']  # This will give you the labels

# data = []


# # labels = []
# # labs = [0,1,2,3,4]
# # cnt = 1000
# # for i, img in enumerate(train_images):
# #     # if i == cnt:
# #     #     break
# #     if train_labels[i] in labs:
# #         data.append(np.array(img).flatten())
# #         labels.append(train_labels[i])
    
    
# for i, img in enumerate(train_images):
#     data.append(np.array(img).flatten())
# labels = train_labels

       
# data = np.stack(data)




# data, labels = datasets.make_blobs(10000, 1000, centers=5)
# with open('data/nytimes-labels.pkl', 'wb') as f:
#     pickle.dump(np.array(labels), f)
# print(data.shape, labels.shape)
cnt = len(set(labels))


model = umap.UMAP(n_neighbors=16)
# # model = cuUMAP(n_neighbors=16, build_algo="nn_descent", build_kwds={'nnd_graph_degree': 32, 'nnd_intermediate_graph_degree': 64, 'nnd_max_iterations': 10, 'nnd_return_distances': True})
# # model = cuUMAP(n_neighbors=16, build_algo="brute_force_knn")

current_time = time.time()
embedding = model.fit_transform(data)
print(f"time: {(time.time() - current_time) * 1000}")

with open('cpu-umap-nytimes-result2.pkl', 'wb') as f:
    pickle.dump(embedding, f)

# with open('cpu-umap-nytimes-result.pkl', 'rb') as file:
#     embedding = pickle.load(file)   
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=0.4)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(cnt+1)-0.5).set_ticks(np.arange(cnt))
plt.title('CPU UMAP projection', fontsize=12)
plt.xlim((-3, 23))
plt.ylim((-10, 16))
plt.savefig("./cpu_umap.png")
