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

import numpy as np

# f = open("jinsolp_out/jinsolp_out_inds_0.txt", "r")
# indices_0_f = f.read()
# indices_0 = np.fromstring(indices_0_f.split("[")[1].split("]")[0], dtype=int, sep=',')
# indices_0 = indices_0.reshape(60000, -1)

# f = open("jinsolp_out/jinsolp_out_inds_after_0.txt", "r")
# indices_0_afteropt_f = f.read()
# indices_0_afteropt = np.fromstring(indices_0_afteropt_f.split("[")[1].split("]")[0], dtype=int, sep=',')
# indices_0_afteropt = indices_0_afteropt.reshape(60000, -1)

# f = open("jinsolp_out/jinsolp_out_inds_1.txt", "r")
# indices_1_f = f.read()
# indices_1 = np.fromstring(indices_1_f.split("[")[1].split("]")[0], dtype=int, sep=',')
# indices_1 = indices_1.reshape(60000, -1)

# f = open("jinsolp_out/jinsolp_out_inds_after_1.txt", "r")
# indices_1_afteropt_f = f.read()
# indices_1_afteropt = np.fromstring(indices_1_afteropt_f.split("[")[1].split("]")[0], dtype=int, sep=',')
# indices_1_afteropt = indices_1_afteropt.reshape(60000, -1)


f = open("jinsolp_out_inds_bfk.txt", "r")
indices_bfk_f = f.read()
indices_bfk = np.fromstring(indices_bfk_f.split("[")[1].split("]")[0], dtype=int, sep=',')
indices_bfk = indices_bfk.reshape(60000, -1)

f = open("jinsolp_out_inds_after_nnd.txt", "r")
indices_nnd_afteropt_f = f.read()
indices_nnd_afteropt = np.fromstring(indices_nnd_afteropt_f.split("[")[1].split("]")[0], dtype=int, sep=',')
indices_nnd_afteropt = indices_nnd_afteropt.reshape(60000, -1)

f = open("jinsolp_out_inds_nnd.txt", "r")
indices_nnd_f = f.read()
indices_nnd = np.fromstring(indices_nnd_f.split("[")[1].split("]")[0], dtype=int, sep=',')
indices_nnd = indices_nnd.reshape(60000, -1)

rows = indices_nnd.shape[0]
cols = indices_nnd.shape[1]

same = 0
for i in range(rows):
    for j in range(cols):
        for k in range(cols):
            if indices_bfk[i][j] == indices_nnd[i][k]:
                same += 1
                break

print(f"recall: {same / (rows * cols)}")


same = 0
for i in range(rows):
    for j in range(cols):
        for k in range(cols):
            if indices_nnd_afteropt[i][j] == indices_nnd[i][k]:
                same += 1
                break

print(f"recall: {same / (rows * cols)}")

same = 0
for i in range(rows):
    for j in range(cols):
        for k in range(cols):
            if indices_nnd_afteropt[i][j] == indices_bfk[i][k]:
                same += 1
                break

print(f"recall: {same / (rows * cols)}")
quit()
same = 0
for i in range(rows):
    for j in range(cols):
        for k in range(cols):
            if indices_0[i][j] == indices_1_afteropt[i][k]:
                same += 1
                break

print(f"recall for bfk: {same / (rows * cols)}")
quit()

same = 0
for i in range(rows):
    for j in range(cols):
        for k in range(cols):
            if indices_1[i][j] == indices_1_afteropt[i][k]:
                same += 1
                break

print(f"recall for nnd: {same / (rows * cols)}")


same = 0
for i in range(rows):
    for j in range(cols):
        for k in range(cols):
            if indices_1[i][j] == indices_0[i][k]:
                same += 1
                break

print(f"recall between bfk & nnd before opt: {same / (rows * cols)}")

same = 0
for i in range(rows):
    for j in range(cols):
        for k in range(cols):
            if indices_1_afteropt[i][j] == indices_0_afteropt[i][k]:
                same += 1
                break

print(f"recall between bfk & nnd after opt: {same / (rows * cols)}")

