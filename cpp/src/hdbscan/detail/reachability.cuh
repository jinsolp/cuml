/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "mst_opt.cuh"

#include <cuml/cluster/hdbscan.hpp>
#include <cuml/common/utils.hpp>
#include <cuml/neighbors/knn.hpp>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/distance.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/neighbors/brute_force.cuh>
#include <raft/neighbors/nn_descent.cuh>
#include <raft/neighbors/nn_descent_types.hpp>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace NNDescent = raft::neighbors::experimental::nn_descent;

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Reachability {

/**
 * Extract core distances from KNN graph. This is essentially
 * performing a knn_dists[:,min_pts]
 * @tparam value_idx data type for integrals
 * @tparam value_t data type for distance
 * @tparam tpb block size for kernel
 * @param[in] knn_dists knn distance array (size n * k)
 * @param[in] min_samples this neighbor will be selected for core distances
 * @param[in] n_neighbors the number of neighbors of each point in the knn graph
 * @param[in] n number of samples
 * @param[out] out output array (size n)
 * @param[in] stream stream for which to order cuda operations
 */
template <typename value_idx, typename value_t, int tpb = 256>
void core_distances(
  value_t* knn_dists, int min_samples, int n_neighbors, size_t n, value_t* out, cudaStream_t stream)
{
  ASSERT(n_neighbors >= min_samples,
         "the size of the neighborhood should be greater than or equal to min_samples");

  auto exec_policy = rmm::exec_policy(stream);

  auto indices = thrust::make_counting_iterator<value_idx>(0);

  thrust::transform(exec_policy, indices, indices + n, out, [=] __device__(value_idx row) {
    return knn_dists[row * n_neighbors + (min_samples - 1)];
  });
}

// Functor to post-process distances by sqrt
// For usage with NN Descent which internally supports L2Expanded only
template <typename value_idx, typename value_t = float>
struct DistancePostProcessSqrt : NNDescent::DistEpilogue<value_idx, value_t> {
  DI value_t operator()(value_t value, value_idx row, value_idx col) const
  {
    return powf(fabsf(value), 0.5);
  }
};

template <typename T>
// out and in can be same (can be done in-place)
CUML_KERNEL void copy_first_k_cols_shift_self(
  T* out, T* in, size_t out_k, size_t in_k, size_t nrows)
{
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    for (size_t i = out_k - 1; i >= 1; i--) {
      out[row * out_k + i] = in[row * in_k + i - 1];
    }
    out[row * out_k] = row;
  }
}

template <typename T>
CUML_KERNEL void copy_first_k_cols_shift_zero(
  T* out, T* in, size_t out_k, size_t in_k, size_t nrows)
{
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    for (size_t i = 1; i < out_k; i++) {
      out[row * out_k + i] = in[row * in_k + i - 1];
    }
    out[row * out_k] = static_cast<T>(0);
  }
}

template <typename T>
// out and in can be same (can be done in-place)
CUML_KERNEL void copy_first_k_cols_shift_core_dists(
  T* out, T* in, T* core_dists, size_t out_k, size_t in_k, size_t nrows)
{
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    for (size_t i = out_k - 1; i >= 1; i--) {
      out[row * out_k + i] = in[row * in_k + i - 1];
    }
    out[row * out_k] = static_cast<T>(core_dists[row]);
  }
}

/**
 * Wraps the brute force knn API, to be used for both training and prediction
 * @tparam value_idx data type for integrals
 * @tparam value_t data type for distance
 * @param[in] handle raft handle for resource reuse
 * @param[in] X input data points (size m * n)
 * @param[out] inds nearest neighbor indices (size n_search_items * k)
 * @param[out] dists nearest neighbor distances (size n_search_items * k)
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[in] search_items array of items to search of dimensionality D (size n_search_items * n)
 * @param[in] n_search_items number of rows in search_items
 * @param[in] k number of nearest neighbors
 * @param[in] metric distance metric to use
 */
template <typename value_idx, typename value_t>
void compute_knn(const raft::handle_t& handle,
                 const value_t* X,
                 value_idx* inds,
                 value_t* dists,
                 size_t m,
                 size_t n,
                 const value_t* search_items,
                 size_t n_search_items,
                 int k,
                 raft::distance::DistanceType metric,
                 Common::GRAPH_BUILD_ALGO build_algo  = Common::GRAPH_BUILD_ALGO::BRUTE_FORCE_KNN,
                 Common::nn_index_params build_params = Common::nn_index_params{})
{
  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();
  // This is temporary. Once faiss is updated, we should be able to
  // pass value_idx through to knn.
  rmm::device_uvector<int64_t> int64_indices(k * n_search_items, stream);

  if (build_algo == Common::GRAPH_BUILD_ALGO::BRUTE_FORCE_KNN) {
    std::vector<value_t*> inputs;
    inputs.push_back(const_cast<value_t*>(X));

    std::vector<int> sizes;
    sizes.push_back(m);

    // perform knn
    brute_force_knn(handle,
                    inputs,
                    sizes,
                    n,
                    const_cast<value_t*>(search_items),
                    n_search_items,
                    int64_indices.data(),
                    dists,
                    k,
                    true,
                    true,
                    metric);
  } else {  // NN_DESCENT
    auto epilogue                 = DistancePostProcessSqrt<int64_t, float>{};
    build_params.return_distances = true;
    RAFT_EXPECTS(static_cast<size_t>(k) <= build_params.graph_degree,
                 "n_neighbors should be smaller than the graph degree computed by nn descent");

    auto dataset = raft::make_host_matrix_view<const float, int64_t>(X, m, n);

    auto graph = NNDescent::build<float, int64_t>(handle, build_params, dataset, epilogue);
    printf("returned from nnd build\n");
    size_t TPB        = 256;
    size_t num_blocks = static_cast<size_t>((m + TPB) / TPB);

    auto indices_d =
      raft::make_device_matrix<int64_t, int64_t>(handle, m, build_params.graph_degree);

    raft::copy(
      indices_d.data_handle(), graph.graph().data_handle(), m * build_params.graph_degree, stream);

    if (graph.distances().has_value()) {
      copy_first_k_cols_shift_zero<float>
        <<<num_blocks, TPB, 0, stream>>>(dists,
                                         graph.distances().value().data_handle(),
                                         static_cast<size_t>(k),
                                         build_params.graph_degree,
                                         m);
    }
    copy_first_k_cols_shift_self<int64_t><<<num_blocks, TPB, 0, stream>>>(int64_indices.data(),
                                                                          indices_d.data_handle(),
                                                                          static_cast<size_t>(k),
                                                                          build_params.graph_degree,
                                                                          m);
  }

  // convert from current knn's 64-bit to 32-bit.
  thrust::transform(exec_policy,
                    int64_indices.data(),
                    int64_indices.data() + int64_indices.size(),
                    inds,
                    [] __device__(int64_t in) -> value_idx { return in; });
}

/*
  @brief Internal function for CPU->GPU interop
         to compute core_dists
*/
template <typename value_idx, typename value_t>
void _compute_core_dists(
  const raft::handle_t& handle,
  const value_t* X,
  value_t* core_dists,
  size_t m,
  size_t n,
  raft::distance::DistanceType metric,
  int min_samples,
  Common::GRAPH_BUILD_ALGO build_algo  = Common::GRAPH_BUILD_ALGO::BRUTE_FORCE_KNN,
  Common::nn_index_params build_params = Common::nn_index_params{})
{
  RAFT_EXPECTS(metric == raft::distance::DistanceType::L2SqrtExpanded,
               "Currently only L2 expanded distance is supported");

  auto stream = handle.get_stream();

  rmm::device_uvector<value_idx> inds(min_samples * m, stream);
  rmm::device_uvector<value_t> dists(min_samples * m, stream);

  // perform knn
  compute_knn(handle,
              X,
              inds.data(),
              dists.data(),
              m,
              n,
              X,
              m,
              min_samples,
              metric,
              build_algo,
              build_params);

  // Slice core distances (distances to kth nearest neighbor)
  core_distances<value_idx>(dists.data(), min_samples, min_samples, m, core_dists, stream);
}

//  Functor to post-process distances into reachability space
template <typename value_idx, typename value_t>
struct ReachabilityPostProcess {
  DI value_t operator()(value_t value, value_idx row, value_idx col) const
  {
    return max(core_dists[col], max(core_dists[row], alpha * value));
  }

  const value_t* core_dists;
  value_t alpha;
};

// Functor to post-process distances into reachability space (Sqrt)
// For usage with NN Descent which internally supports L2Expanded only
template <typename value_idx, typename value_t = float>
struct ReachabilityPostProcessSqrt : NNDescent::DistEpilogue<value_idx, value_t> {
  ReachabilityPostProcessSqrt(value_t* core_dists_, value_t alpha_)
    : NNDescent::DistEpilogue<value_idx, value_t>(),
      core_dists(core_dists_),
      alpha(alpha_),
      value_t_max(std::numeric_limits<value_t>::max()){};

  __device__ value_t operator()(value_t value, value_idx row, value_idx col) const
  {
    if (cluster_indices == nullptr) {
      return max(core_dists[col], max(core_dists[row], powf(fabsf(alpha * value), 0.5)));
    } else {
      if (row < num_data_in_cluster && col < num_data_in_cluster) {
        return max(core_dists[cluster_indices[col]],
                   max(core_dists[cluster_indices[row]], powf(fabsf(alpha * value), 0.5)));
      } else {
        return value_t_max;
      }
    }
  }

  __host__ void preprocess_for_batch(value_idx* cluster_indices_, size_t num_data_in_cluster_)
  {
    cluster_indices     = cluster_indices_;
    num_data_in_cluster = num_data_in_cluster_;
  }

  const value_t* core_dists;
  value_t alpha;
  value_t value_t_max;
  value_idx* cluster_indices = nullptr;
  size_t num_data_in_cluster = 0;
};

template <typename value_idx>
struct CustomComparator {
  __host__ __device__ bool operator()(const thrust::tuple<float, value_idx>& lhs,
                                      const thrust::tuple<float, value_idx>& rhs) const
  {
    if (thrust::get<0>(lhs) != thrust::get<0>(rhs)) {
      return thrust::get<0>(lhs) < thrust::get<0>(rhs);
    }
    return thrust::get<1>(lhs) < thrust::get<1>(rhs);
  }
};

template <typename value_t>
float calculate_mutual_reach_dist(
  const raft::handle_t& handle, const value_t* X, int i, int j, float core_dist, size_t dim)
{
  auto x_i = raft::make_host_vector_view<const value_t, int64_t>(X + i * dim, dim);
  auto x_j = raft::make_host_vector_view<const value_t, int64_t>(X + j * dim, dim);

  float x_i_norm = 0;
  float x_j_norm = 0;
  float dot      = 0;
  for (int d = 0; d < dim; d++) {
    x_i_norm += x_i(d) * x_i(d);
    x_j_norm += x_j(d) * x_j(d);
    dot += x_i(d) * x_j(d);
  }

  return std::max((float)(std::sqrt(x_i_norm + x_j_norm - 2.0 * dot)), core_dist);
}

template <typename KeyType, typename ValueType>
struct KeyValuePair {
  KeyType key;
  ValueType value;
};

template <typename KeyType, typename ValueType>
struct CustomKeyComparator {
  __device__ bool operator()(const KeyValuePair<KeyType, ValueType>& a,
                             const KeyValuePair<KeyType, ValueType>& b) const
  {
    if (a.key == b.key) { return a.value < b.value; }
    return a.key < b.key;
  }
};

template <typename value_idx, int BLOCK_SIZE, int ITEMS_PER_THREAD>
CUML_KERNEL void sort_by_key(float* out_dists,
                             value_idx* out_inds,
                             size_t graph_degree,
                             size_t nrows)
{
  size_t row = blockIdx.x;
  typedef cub::BlockMergeSort<KeyValuePair<float, value_idx>, BLOCK_SIZE, ITEMS_PER_THREAD>
    BlockMergeSortType;
  __shared__ typename cub::BlockMergeSort<KeyValuePair<float, value_idx>,
                                          BLOCK_SIZE,
                                          ITEMS_PER_THREAD>::TempStorage tmpSmem;

  if (row < nrows) {
    KeyValuePair<float, value_idx> threadKeyValuePair[ITEMS_PER_THREAD];

    // load key values
    size_t arrIdxBase = row * graph_degree;
    size_t idxBase    = static_cast<size_t>(threadIdx.x) * static_cast<size_t>(ITEMS_PER_THREAD);
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t colId = idxBase + static_cast<size_t>(i);
      if (colId < graph_degree) {
        threadKeyValuePair[i].key   = out_dists[arrIdxBase + colId];
        threadKeyValuePair[i].value = out_inds[arrIdxBase + colId];
      } else {
        threadKeyValuePair[i].key   = std::numeric_limits<float>::max();
        threadKeyValuePair[i].value = std::numeric_limits<value_idx>::max();
      }
    }

    __syncthreads();

    BlockMergeSortType(tmpSmem).Sort(threadKeyValuePair, CustomKeyComparator<float, value_idx>{});

    // load back to global mem
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t colId = idxBase + static_cast<size_t>(i);
      if (colId < graph_degree) {
        out_dists[arrIdxBase + colId] = threadKeyValuePair[i].key;
        out_inds[arrIdxBase + colId]  = threadKeyValuePair[i].value;
      }
    }
  }
}

/**
 * Given core distances, Fuses computations of L2 distances between all
 * points, projection into mutual reachability space, and k-selection.
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[out] out_inds  output indices array (size m * k)
 * @param[out] out_dists output distances array (size m * k)
 * @param[in] X input data points (size m * n)
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[in] k neighborhood size (includes self-loop)
 * @param[in] core_dists array of core distances (size m)
 */
template <typename value_idx, typename value_t>
void mutual_reachability_knn_l2(
  const raft::handle_t& handle,
  value_idx* out_inds,
  value_t* out_dists,
  const value_t* X,
  size_t m,
  size_t n,
  int k,
  value_t* core_dists,
  value_t alpha,
  Common::GRAPH_BUILD_ALGO build_algo  = Common::GRAPH_BUILD_ALGO::BRUTE_FORCE_KNN,
  Common::nn_index_params build_params = Common::nn_index_params{},
  bool approx_mst                      = false)
{
  // Create a functor to postprocess distances into mutual reachability space
  // Note that we can't use a lambda for this here, since we get errors like:
  // `A type local to a function cannot be used in the template argument of the
  // enclosing parent function (and any parent classes) of an extended __device__
  // or __host__ __device__ lambda`

  if (build_algo == Common::GRAPH_BUILD_ALGO::BRUTE_FORCE_KNN) {
    auto epilogue = ReachabilityPostProcess<value_idx, value_t>{core_dists, alpha};
    auto X_view   = raft::make_device_matrix_view(X, m, n);
    std::vector<raft::device_matrix_view<const value_t, size_t>> index = {X_view};

    raft::neighbors::brute_force::knn<value_idx, value_t>(
      handle,
      index,
      X_view,
      raft::make_device_matrix_view(out_inds, m, static_cast<size_t>(k)),
      raft::make_device_matrix_view(out_dists, m, static_cast<size_t>(k)),
      // TODO: expand distance metrics to support more than just L2 distance
      // https://github.com/rapidsai/cuml/issues/5301
      raft::distance::DistanceType::L2SqrtExpanded,
      std::make_optional<float>(2.0f),
      std::nullopt,
      epilogue);

  } else {
    auto epilogue = ReachabilityPostProcessSqrt<value_idx, value_t>{core_dists, alpha};
    build_params.return_distances = true;
    // build_params.graph_degree     = (size_t)((build_params.graph_degree * 1.5 + 31) / 32) * 32;
    printf("build params graph degree %lu\n", build_params.graph_degree);
    RAFT_EXPECTS(static_cast<size_t>(k) <= build_params.graph_degree,
                 "n_neighbors should be smaller than the graph degree computed by nn descent");

    auto dataset = raft::make_host_matrix_view<const value_t, int64_t>(X, m, n);
    auto graph   = NNDescent::build<value_t, value_idx>(handle, build_params, dataset, epilogue);

    size_t TPB        = 256;
    size_t num_blocks = static_cast<size_t>((m + TPB) / TPB);

    auto indices_d =
      raft::make_device_matrix<value_idx, value_idx>(handle, m, build_params.graph_degree);

    raft::copy(indices_d.data_handle(),
               graph.graph().data_handle(),
               m * build_params.graph_degree,
               handle.get_stream());

    RAFT_EXPECTS(graph.distances().has_value(),
                 "return_distances for nn descent should be set to true to be used for HDBSCAN");

    auto start = raft::curTimeMillis();

    if (approx_mst) {
      printf("\tdoing mst optimize\n");
      copy_first_k_cols_shift_core_dists<float>
        <<<num_blocks, TPB, 0, handle.get_stream()>>>(graph.distances().value().data_handle(),
                                                      graph.distances().value().data_handle(),
                                                      core_dists,
                                                      build_params.graph_degree,
                                                      build_params.graph_degree,
                                                      m);
      copy_first_k_cols_shift_self<value_idx>
        <<<num_blocks, TPB, 0, handle.get_stream()>>>(indices_d.data_handle(),
                                                      indices_d.data_handle(),
                                                      build_params.graph_degree,
                                                      build_params.graph_degree,
                                                      m);

      auto start2 = raft::curTimeMillis();

      auto new_inds  = raft::make_host_matrix<value_idx, int64_t>(m, k);
      auto new_dists = raft::make_host_matrix<float, int64_t>(m, k);

      auto knn_dists = raft::make_host_matrix<float, int64_t>(m, build_params.graph_degree);
      raft::copy(graph.graph().data_handle(),
                 indices_d.data_handle(),
                 m * build_params.graph_degree,
                 handle.get_stream());
      auto knn_inds = raft::make_host_matrix_view<value_idx, int64_t>(
        graph.graph().data_handle(), m, build_params.graph_degree);  // reuse memory
      raft::copy(knn_dists.data_handle(),
                 graph.distances().value().data_handle(),
                 m * build_params.graph_degree,
                 handle.get_stream());

      optimize(handle, knn_inds, new_inds.view(), true);

      auto core_dists_h = raft::make_host_vector<float, int64_t>(m);
      raft::copy(core_dists_h.data_handle(), core_dists, m, handle.get_stream());

#pragma omp parallel for
      for (size_t i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
          value_idx curr_idx = new_inds.data_handle()[i * k + j];
          bool found         = false;
          for (int l = 0; l < build_params.graph_degree; l++) {
            if (knn_inds.data_handle()[i * build_params.graph_degree + l] == curr_idx) {
              new_dists.data_handle()[i * k + j] =
                knn_dists.data_handle()[i * build_params.graph_degree + l];
              found = true;
              break;
            }
          }
          if (!found) {
            new_dists.data_handle()[i * k + j] =
              calculate_mutual_reach_dist<value_t>(handle, X, i, curr_idx, core_dists_h(i), n);
          }
        }
      }

      raft::copy(out_inds, new_inds.data_handle(), m * k, handle.get_stream());
      raft::copy(out_dists, new_dists.data_handle(), m * k, handle.get_stream());

      auto end2 = raft::curTimeMillis();
      printf("\t\tdoing mst optimize first part time %d\n", end2 - start2);

      // // thrust::host_vector<value_idx> h_vals_vec(inds.data(), inds.data() + m*min_samples);
      // // thrust::host_vector<float> h_keys_vec(dists.data(), dists.data() + m*min_samples);
      start2 = raft::curTimeMillis();

      if (k <= 128) {
        sort_by_key<value_idx, 32, 4><<<m, 32, 0, handle.get_stream()>>>(out_dists, out_inds, k, m);
      }
      handle.sync_stream();
      // std::vector<value_idx> std_vector_value(new_inds.data_handle(), new_inds.data_handle() + m
      // * k); std::vector<float> std_vector_key(new_dists.data_handle(), new_dists.data_handle() +
      // m * k);

      // thrust::host_vector<value_idx> h_vals_vec(std_vector_value.begin(),
      // std_vector_value.end()); thrust::host_vector<float> h_keys_vec(std_vector_key.begin(),
      // std_vector_key.end());

      // auto tuple_begin =
      //   thrust::make_zip_iterator(thrust::make_tuple(h_keys_vec.begin(), h_vals_vec.begin()));
      // auto tuple_end =
      //   thrust::make_zip_iterator(thrust::make_tuple(h_keys_vec.end(), h_vals_vec.end()));

      // for (size_t i = 0; i < m; i++) {
      //   thrust::sort(tuple_begin + i * k, tuple_begin + (i + 1) * k,
      //   CustomComparator<value_idx>());
      // }

      // int cnt = 0;
      // for (auto it = tuple_begin; it != tuple_end; ++it) {
      //   auto [val1, val2]                         = *it;  // Unpack the tuple
      //   new_inds((int)(cnt / k), (int)(cnt % k))  = val2;
      //   new_dists((int)(cnt / k), (int)(cnt % k)) = val1;
      //   cnt++;
      // }

      // // copy back
      // raft::copy(out_inds, new_inds.data_handle(), m * k, handle.get_stream());
      // raft::copy(out_dists, new_dists.data_handle(), m * k, handle.get_stream());

      end2 = raft::curTimeMillis();
      printf("\t\tdoing mst optimize sorting part %d\n", end2 - start2);

    } else {
      printf("\tnot doing mst optimize\n");
      copy_first_k_cols_shift_core_dists<float>
        <<<num_blocks, TPB, 0, handle.get_stream()>>>(out_dists,
                                                      graph.distances().value().data_handle(),
                                                      core_dists,
                                                      k,
                                                      build_params.graph_degree,
                                                      m);
      copy_first_k_cols_shift_self<value_idx><<<num_blocks, TPB, 0, handle.get_stream()>>>(
        out_inds, indices_d.data_handle(), k, build_params.graph_degree, m);
    }
    auto end = raft::curTimeMillis();
    printf("time to do mst postprocessing (or maybe not) %d\n", end - start);
  }
}

/**
 * Constructs a mutual reachability graph, which is a k-nearest neighbors
 * graph projected into mutual reachability space using the following
 * function for each data point, where core_distance is the distance
 * to the kth neighbor: max(core_distance(a), core_distance(b), d(a, b))
 *
 * Unfortunately, points in the tails of the pdf (e.g. in sparse regions
 * of the space) can have very large neighborhoods, which will impact
 * nearby neighborhoods. Because of this, it's possible that the
 * radius for points in the main mass, which might have a very small
 * radius initially, to expand very large. As a result, the initial
 * knn which was used to compute the core distances may no longer
 * capture the actual neighborhoods after projection into mutual
 * reachability space.
 *
 * For the experimental version, we execute the knn twice- once
 * to compute the radii (core distances) and again to capture
 * the final neighborhoods. Future iterations of this algorithm
 * will work improve upon this "exact" version, by using
 * more specialized data structures, such as space-partitioning
 * structures. It has also been shown that approximate nearest
 * neighbors can yield reasonable neighborhoods as the
 * data sizes increase.
 *
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[in] X input data points (size m * n)
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[in] metric distance metric to use
 * @param[in] k neighborhood size
 * @param[in] min_samples this neighborhood will be selected for core distances
 * @param[in] alpha weight applied when internal distance is chosen for
 *            mutual reachability (value of 1.0 disables the weighting)
 * @param[out] indptr CSR indptr of output knn graph (size m + 1)
 * @param[out] core_dists output core distances array (size m)
 * @param[out] out COO object, uninitialized on entry, on exit it stores the
 *             (symmetrized) maximum reachability distance for the k nearest
 *             neighbors.
 */
template <typename value_idx, typename value_t>
void mutual_reachability_graph(
  const raft::handle_t& handle,
  const value_t* X,
  size_t m,
  size_t n,
  raft::distance::DistanceType metric,
  int min_samples,
  value_t alpha,
  value_idx* indptr,
  value_t* core_dists,
  raft::sparse::COO<value_t, value_idx>& out,
  Common::GRAPH_BUILD_ALGO build_algo  = Common::GRAPH_BUILD_ALGO::BRUTE_FORCE_KNN,
  Common::nn_index_params build_params = Common::nn_index_params{},
  bool approx_mst                      = false)
{
  RAFT_EXPECTS(metric == raft::distance::DistanceType::L2SqrtExpanded,
               "Currently only L2 expanded distance is supported");

  auto stream      = handle.get_stream();
  auto exec_policy = handle.get_thrust_policy();

  rmm::device_uvector<value_idx> coo_rows(min_samples * m, stream);
  rmm::device_uvector<value_idx> inds(min_samples * m, stream);
  rmm::device_uvector<value_t> dists(min_samples * m, stream);

  // perform knn
  compute_knn(handle,
              X,
              inds.data(),
              dists.data(),
              m,
              n,
              X,
              m,
              min_samples,
              metric,
              build_algo,
              build_params);

  // Slice core distances (distances to kth nearest neighbor)
  core_distances<value_idx>(dists.data(), min_samples, min_samples, m, core_dists, stream);
  printf("returned from core dists\n");
  /**
   * Compute L2 norm
   */
  mutual_reachability_knn_l2(handle,
                             inds.data(),
                             dists.data(),
                             X,
                             m,
                             n,
                             min_samples,
                             core_dists,
                             (value_t)1.0 / alpha,
                             build_algo,
                             build_params,
                             approx_mst);

  // self-loops get max distance
  auto coo_rows_counting_itr = thrust::make_counting_iterator<value_idx>(0);
  thrust::transform(exec_policy,
                    coo_rows_counting_itr,
                    coo_rows_counting_itr + (m * min_samples),
                    coo_rows.data(),
                    [min_samples] __device__(value_idx c) -> value_idx { return c / min_samples; });

  raft::sparse::linalg::symmetrize(
    handle, coo_rows.data(), inds.data(), dists.data(), m, m, min_samples * m, out);

  raft::sparse::convert::sorted_coo_to_csr(out.rows(), out.nnz, indptr, m + 1, stream);

  // self-loops get max distance
  auto transform_in =
    thrust::make_zip_iterator(thrust::make_tuple(out.rows(), out.cols(), out.vals()));

  thrust::transform(exec_policy,
                    transform_in,
                    transform_in + out.nnz,
                    out.vals(),
                    [=] __device__(const thrust::tuple<value_idx, value_idx, value_t>& tup) {
                      return thrust::get<0>(tup) == thrust::get<1>(tup)
                               ? std::numeric_limits<value_t>::max()
                               : thrust::get<2>(tup);
                    });
}

};  // end namespace Reachability
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML
