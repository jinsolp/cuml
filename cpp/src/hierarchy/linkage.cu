/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <cuml/cluster/linkage.hpp>

#include <raft/cluster/single_linkage.cuh>
#include <raft/core/handle.hpp>

namespace ML {

void single_linkage_pairwise(const raft::handle_t& handle,
                             const float* X,
                             size_t m,
                             size_t n,
                             raft::cluster::linkage_output<int>* out,
                             cuvs::distance::DistanceType metric,
                             int n_clusters)
{
  raft::cluster::single_linkage<int, float, raft::cluster::LinkageDistance::PAIRWISE>(
    handle, X, m, n, static_cast<raft::distance::DistanceType>(metric), out, 0, n_clusters);
}

void single_linkage_neighbors(const raft::handle_t& handle,
                              const float* X,
                              size_t m,
                              size_t n,
                              raft::cluster::linkage_output<int>* out,
                              cuvs::distance::DistanceType metric,
                              int c,
                              int n_clusters)
{
  raft::cluster::single_linkage<int, float, raft::cluster::LinkageDistance::KNN_GRAPH>(
    handle, X, m, n, static_cast<raft::distance::DistanceType>(metric), out, c, n_clusters);
}

struct distance_graph_impl_int_float
  : public raft::cluster::detail::
      distance_graph_impl<raft::cluster::LinkageDistance::PAIRWISE, int, float> {};
struct distance_graph_impl_int_double
  : public raft::cluster::detail::
      distance_graph_impl<raft::cluster::LinkageDistance::PAIRWISE, int, double> {};

};  // end namespace ML
