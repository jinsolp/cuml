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

import matplotlib.pyplot as plt
import numpy as np



nvidia_green = "#76B900"
# plt.figure(figsize=(10, 4))

fig, axs = plt.subplots(1, 4, figsize=(10, 2.5))
get_graph = np.array([545, 50026, 538769,3023892])/1000
e2e = np.array([941, 53836, 541961, 3031010])/1000

labels=["Building KNN graph", "Other"]

axs[0].pie([get_graph[0], e2e[0] - get_graph[0]], autopct='%1.1f%%', colors=[nvidia_green, "orange"])
axs[1].pie([get_graph[1], e2e[1] - get_graph[1]], autopct='%1.1f%%', colors=[nvidia_green, "orange"])
axs[2].pie([get_graph[2], e2e[2] - get_graph[2]], autopct='%1.1f%%', colors=[nvidia_green, "orange"])
axs[3].pie([get_graph[3], e2e[3] - get_graph[3]], autopct='%1.1f%%', colors=[nvidia_green, "orange"])

axs[0].text(0, -1.4, 'Mnist\n(60K, 784)', fontsize=12, ha='center', va='center')
axs[1].text(0, -1.4, "Sift\n(1M, 128)", fontsize=12, ha='center', va='center')
axs[2].text(0, -1.4, "Gist\n(1M, 960)", fontsize=12, ha='center', va='center')
axs[3].text(0, -1.4, "Amazon food review\n(5M,384)", fontsize=12, ha='center', va='center')
# plt.yscale("log")
# # plt.ylim(0.0, 1000)

# plt.bar(x, get_graph, label="Building KNN graph")
# plt.bar(x, e2e-get_graph, label="other", bottom=get_graph)

# plt.legend(fontsize=8)

# plt.xticks(x, labels=["Mnist\n(60K, 784)", "Sift\n(1M, 128)","Gist\n(1M, 960)", "Amazon food\nreview (5M,384)"])

axs[0].legend(labels, loc=(-0.5,0.93))
plt.suptitle("Proportion of building KNN graph\nof running UMAP end to end", fontsize=15)
plt.savefig("./blog_breakdown.png")

quit()


fig, ax1 = plt.subplots()
x = np.array([0,1,2,3,4,5,6])
get_graph = np.array([22126, 50301, 49990, 43945, 39919, 48919, 64783])/1000
e2e = np.array([28418, 56602, 56323, 50261, 45467, 55273, 71194])/1000
trust = np.array([0.9229, 0.9251, 0.9237, 0.9225, 0.9210, 0.9199, 0.9221])

ax1.bar(x, get_graph, label="Building KNN graph")
ax1.bar(x, e2e-get_graph, label="other", bottom=get_graph)
ax1.plot(0,0, label="trustworthiness score", color="green")

ax1.legend(fontsize=8)
ax2 = ax1.twinx()
ax2.plot(x, trust, marker='o', markersize=5, color="green")

plt.xticks(x, labels=["No batch", "2", "4", "8", "16", "32", "64"])

ax2.set_ylabel("Trustworthiness Score", color="green")
ax2.set_ylim(0.75, 1.0)
# plt.ylim(0.75, 1.0)

plt.title("Time to run cuML UMAP and trustworthiness on (5M, 384) data\nfor different number of batches")
ax1.set_ylabel("Time (s)")
plt.savefig("./umap_result9.png")

quit()


barwidth = 0.3
x = np.array([0,1,2])
graph_build = np.array([26753, 93531, 144035])/1000
condense_hierarchy = np.array([63751, 422247, 1571399])/1000
total = np.array([93371, 538729, 1774645])/1000

graph_build_bfk = np.array([250382, 2839268, 3906810])/1000
condense_hierarchy_bfk = np.array([67664, 424282, 1573713])/1000
total_bfk = np.array([322245, 3291231, 5535716])/1000

color_one = "#FFA500"

plt.bar(x-barwidth/2, graph_build, label = "Building KNN graph (NN Descent)", width = barwidth)
plt.bar(x[:3]+barwidth/2, graph_build_bfk, label = "Building KNN graph (brute force)", width = barwidth, color="purple")
plt.bar(x-barwidth/2, condense_hierarchy, label = "Condense Hierarchy", width = barwidth, bottom = graph_build, color=color_one)
plt.bar(x-barwidth/2, total - (graph_build + condense_hierarchy), label = "Other", width = barwidth, bottom = graph_build + condense_hierarchy, color="green")
# plt.text(3 - barwidth/2, total[2], '20\nclusters', fontsize=8, color='black', ha='center', va='center')
# plt.text(3 + barwidth/2, 90, 'OOM', fontsize=8, color="black", ha='center', va='center')

plt.bar(x[:3]+barwidth/2, condense_hierarchy_bfk, width = barwidth, bottom = graph_build_bfk, color=color_one)
plt.bar(x[:3]+barwidth/2, total_bfk - (graph_build_bfk + condense_hierarchy_bfk),  width = barwidth, bottom = graph_build_bfk + condense_hierarchy_bfk, color="green")

plt.text(0 - barwidth/2, total[0] + 100, 'x3.4', fontsize=10, color='red', ha='center', va='center')
plt.text(1 - barwidth/2, total[1] + 100, 'x6.1', fontsize=10, color='red', ha='center', va='center')
plt.text(2 - barwidth/2, total[2] + 100, 'x3.1', fontsize=10, color='red', ha='center', va='center')


plt.ylabel("Time (s)")
plt.title("Breakdown of running cuML HDBSCAN")
plt.legend()
plt.xticks(x, labels=["Gist\n(1M, 960)", "Amazon food\nreview (5M,384)","Deep Image\n(10M, 96)"])


plt.savefig("./hdbscan_result4.png")
quit()


barwidth = 0.3
x = np.array([0,1,2, 3])
nnd = np.array([93371, 538729, 1774645, 53189480])/1000
bfk = np.array([322245, 3291231, 5545716])/1000

color_one = "#76B900"
color_two = "#008564"

fig, ax1 = plt.subplots()
ax1.set_yscale("log")

ax1.bar(x-barwidth/2, nnd, label = "Run HDBSCAN with NN Descent", width = barwidth, color = color_one)
ax1.bar(x[:3]+barwidth/2, bfk, label = "Run HDBSCAN with Brute Force", width = barwidth, color=color_two)

ax1.text(0 - barwidth/2, nnd[0] + 10, 'x3.4', fontsize=8, color='red', ha='center', va='center')
ax1.text(1 - barwidth/2, nnd[1] + 50, 'x6.1', fontsize=8, color='red', ha='center', va='center')
ax1.text(2 - barwidth/2, nnd[2] + 200, 'x3.1', fontsize=8, color='red', ha='center', va='center')
ax1.text(3 - barwidth/2, nnd[2], '20\nclusters', fontsize=8, color='black', ha='center', va='center')
ax1.text(3 + barwidth/2, 90, 'OOM', fontsize=8, color=color_two, ha='center', va='center')
ax1.set_xlim((-0.5, 3.5))


ax1.set_ylabel("Time (s)", color="green")
ax1.set_xticks(ticks=x)
ax1.set_xticklabels(["Gist\n(1M, 960)", "Amazon food\nreview (5M,384)","Deep Image\n(10M, 96)", "Wiki-all subsample\n(50M, 768)"])

ax1.set_yscale("log")
ax1.legend(fontsize=8)

plt.title("Running cuML HDBSCAN end to end (log scale)")

plt.savefig("./hdbscan_result3.png")

quit()




fig, ax1 = plt.subplots()
ax1.set_yscale("log")

ax1.bar(x - barwidth/2, nnd, width=barwidth, color=color_one, label="Time to run UMAP with NN Descent")
ax1.bar(x[:4] + barwidth/2, bfk, width=barwidth, color=color_two, label="Time to run UMAP with Brute Force")
ax1.set_xticks(ticks=x)
ax1.set_xticklabels(["Gist\n(1M, 960)", "Amazon food\nreview\n(5M,384)","Deep Image\n(10M, 96)", "Amazon Electronics\nreview\n(20M, 384)", "Wiki-all\nsubsample\n(50M, 768)"], fontsize=8)
ax1.text(0 - barwidth/2, nnd[0] + 1, 'x21.6', fontsize=8, color='red', ha='center', va='center')
ax1.text(1 - barwidth/2, nnd[1] + 5, 'x64.4', fontsize=8, color='red', ha='center', va='center')
ax1.text(2 - barwidth/2, nnd[2] + 7, 'x40.6', fontsize=8, color='red', ha='center', va='center')
ax1.text(3 - barwidth/2 - 0.05, nnd[3] + 25, 'x311.8', fontsize=8, color='red', ha='center', va='center')
ax1.text(4 - barwidth/2, nnd[2], '5\nclusters', fontsize=8, color='black', ha='center', va='center')
ax1.text(4 + barwidth/2, 8, 'OOM', fontsize=8, color=color_two, ha='center', va='center')
ax1.set_xlim((-0.5, 4.5))
ax1.tick_params(axis='y', labelcolor="green")

plot_color="#4b0082"
ax1.plot([0], [0], color=plot_color, label="trustworthiness scores")
ax1.legend(fontsize=8)

ax1.set_ylabel("Time (s)", color="green")


ax2 = ax1.twinx()
ax2.set_ylim((0.7, 1))
ax2.set_ylabel("Trustworthiness Score", color=plot_color)
ax2.tick_params(axis='y', labelcolor=plot_color)
ax2.set_yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
ax2.plot([0-barwidth/1.5, 0 + barwidth/1.5], [0.7807, 0.7832], marker='o', markersize=5, color=plot_color)
ax2.plot([1-barwidth/1.5, 1 + barwidth/1.5], [0.9236, 0.9149], marker='o', markersize=5, color=plot_color)
ax2.plot([2-barwidth/1.5, 2 + barwidth/1.5], [0.8929, 0.8829], marker='o', markersize=5, color=plot_color)
ax2.plot([3-barwidth/1.5, 3 + barwidth/1.5], [0.9156, 0.9030], marker='o', markersize=5, color=plot_color)
ax2.plot([4-barwidth/2], [0.8122], marker='o', markersize=5, color=plot_color)

plt.title("Time to run UMAP in RAPIDS cuML\nusing NN Descent VS Brute Force KNN (log scale)")
plt.savefig("./umap_result_tmp.png")
quit()



barwidth = 0.5
x = np.array([0,1,2])
graph_build = np.array([250382, 2839268, 3906810])/1000
condense_hierarchy = np.array([67664, 424282, 1573713])/1000
total = np.array([322245, 3291231, 5545716])/1000

plt.bar(x, graph_build, label = "Building KNN graph (brute force)", width = barwidth)
plt.bar(x, condense_hierarchy, label = "Condense Hierarchy", width = barwidth, bottom = graph_build)
plt.bar(x, total - (graph_build + condense_hierarchy), label = "Other", width = barwidth, bottom = graph_build + condense_hierarchy)


plt.ylabel("Time (s)")
plt.title("Breakdown of running cuML HDBSCAN")
plt.legend()
plt.xticks(x, labels=["Gist\n(1M, 960)", "Amazon food\nreview (5M,384)","Deep Image\n(10M, 96)"])


plt.savefig("./hdbscan_result1.png")
quit()


plt.figure(figsize=(6, 6))
barwidth = 0.5
x = np.array([0,1,2,3])

labels = ["preprocessing", "allocating memory", "build iterations", "other"]
gist = np.array([937, 1532, 10777, 13926]) / 1000
# plt.subplot(1, 5, 1)
# plt.bar([0], [gist[0]], label=labels[0], width = barwidth)
# plt.bar([0], [gist[1]], label=labels[1], bottom=np.sum(gist[:1]), width = barwidth)
# plt.bar([0], [gist[2]], label=labels[2], bottom=np.sum(gist[:2]), width = barwidth)
# plt.bar([0], [gist[3] - np.sum(gist[:3])], label=labels[3], bottom=np.sum(gist[:3]), width = barwidth)
# plt.xlim((-0.5, 0.5))

food = np.array([2810, 5988, 36880, 50569]) / 1000
# plt.subplot(1, 5, 2)
# plt.bar([0], [food[0]], label=labels[0], width = barwidth)
# plt.bar([0], [food[1]], label=labels[1], bottom=np.sum(food[:1]), width = barwidth)
# plt.bar([0], [food[2]], label=labels[2], bottom=np.sum(food[:2]), width = barwidth)
# plt.bar([0], [food[3] - np.sum(food[:3])], label=labels[3], bottom=np.sum(food[:3]), width = barwidth)
# plt.xlim((-0.5, 0.5))

deep = np.array([1745, 11401, 52659, 71295]) / 1000
# plt.subplot(1, 5, 3)
# plt.bar([0], [deep[0]], label=labels[0], width = barwidth)
# plt.bar([0], [deep[1]], label=labels[1], bottom=np.sum(deep[:1]), width = barwidth)
# plt.bar([0], [deep[2]], label=labels[2], bottom=np.sum(deep[:2]), width = barwidth)
# plt.bar([0], [deep[3] - np.sum(deep[:3])], label=labels[3], bottom=np.sum(deep[:3]), width = barwidth)
# plt.xlim((-0.5, 0.5))

elec = np.array([6107, 17010, 145358, 189787]) / 1000
# plt.subplot(1, 5, 4)
# plt.bar([0], [elec[0]], label=labels[0], width = barwidth)
# plt.bar([0], [elec[1]], label=labels[1], bottom=np.sum(elec[:1]), width = barwidth)
# plt.bar([0], [elec[2]], label=labels[2], bottom=np.sum(elec[:2]), width = barwidth)
# plt.bar([0], [elec[3] - np.sum(elec[:3])], label=labels[3], bottom=np.sum(elec[:3]), width = barwidth)
# plt.xlim((-0.5, 0.5))

wiki = np.array([29622, 38364, 460301, 582315]) / 1000
# plt.subplot(1, 5, 5)
# plt.bar([0], [wiki[0]], label=labels[0], width = barwidth)
# plt.bar([0], [wiki[1]], label=labels[1], bottom=np.sum(wiki[:1]), width = barwidth)
# plt.bar([0], [wiki[2]], label=labels[2], bottom=np.sum(wiki[:2]), width = barwidth)
# plt.bar([0], [wiki[3] - np.sum(wiki[:3])], label=labels[3], bottom=np.sum(wiki[:3]), width = barwidth)
# plt.xlim((-0.5, 0.5))

all_values = np.stack([gist, food, deep, elec])
print(np.sum(all_values[:, :2], axis=1))
plt.bar(x, all_values[:, 0].reshape(-1), label = labels[0], width = barwidth)
plt.bar(x, all_values[:, 1].reshape(-1), label = labels[1], width = barwidth, bottom = np.sum(all_values[:, :1], axis=1))
plt.bar(x, all_values[:, 2].reshape(-1), label = labels[2], width = barwidth, bottom = np.sum(all_values[:, :2], axis=1))
plt.bar(x, all_values[:, 3].reshape(-1) - np.sum(all_values[:, :3], axis=1), label = labels[3], width = barwidth, bottom = np.sum(all_values[:, :3], axis=1))

# plt.subplots_adjust(left=0.125, bottom=None, right=0.9, top=None, wspace=0.6, hspace=None)
plt.legend()
plt.title("Breakdown of building knn graph\nusing batched NND")
plt.ylabel("Time (s)")
plt.xticks(x, labels=["Gist\n(1M, 960)", "Amazon food\nreview\n(5M,384)","Deep Image\n(10M, 96)", "Amazon Electronics\nreview\n(20M, 384)"])
plt.savefig("./umap_result8.png")

quit()

x = np.array([0,1,2,3])
gist = np.array([0.7807, 0.7807, 0.7834, 0.7834])
food = np.array([0.9251, 0.9263, 0.9237, 0.9227])
deep = np.array([0.9012, 0.8983, 0.8987, 0.8917])
elec = np.array([0.9146, 0.9152, 0.9116, 0.9138])
wiki = np.array([0.8184, 0.8182, 0.8127])

plt.plot(x, gist, label="Gist (1M, 960)", marker='o', markersize=5)
plt.plot(x, food, label="Amazon food review (5M,384)", marker='o', markersize=5)
plt.plot(x, deep, label="Deep Image (10M, 96)", marker='o', markersize=5)
plt.plot(x, elec, label="Amazon Electronics review (20M, 384)", marker='o', markersize=5)
plt.plot(x[1:], wiki, label="Wiki-all subsample (50M, 768))", marker='o', markersize=5)

plt.xticks(x, labels=["No batch", "5 batches", "10 batchs", "15 batches"])
plt.legend(fontsize=8)
plt.ylim(0.75, 1.0)
plt.title("Change of UMAP Trustworthiness Score\nDepending on number of batches used")
plt.ylabel("Trustworthiness Score")
plt.savefig("./umap_result7.png")

quit()

barwidth=0.3
x = np.array([0,1,2,3, 4])

color_one = "#76B900"
color_two = "#008564"


nnd = np.array([9933, 34030, 53411, 122979, 575165])/1000
bfk = np.array([214466, 2191373, 2170869, 38350718])/1000

fig, ax1 = plt.subplots()
ax1.set_yscale("log")

ax1.bar(x - barwidth/2, nnd, width=barwidth, color=color_one, label="Time to run UMAP with NN Descent")
ax1.bar(x[:4] + barwidth/2, bfk, width=barwidth, color=color_two, label="Time to run UMAP with Brute Force")
ax1.set_xticks(ticks=x)
ax1.set_xticklabels(["Gist\n(1M, 960)", "Amazon food\nreview\n(5M,384)","Deep Image\n(10M, 96)", "Amazon Electronics\nreview\n(20M, 384)", "Wiki-all\nsubsample\n(50M, 768)"], fontsize=8)
ax1.text(0 - barwidth/2, nnd[0] + 1, 'x21.6', fontsize=8, color='red', ha='center', va='center')
ax1.text(1 - barwidth/2, nnd[1] + 5, 'x64.4', fontsize=8, color='red', ha='center', va='center')
ax1.text(2 - barwidth/2, nnd[2] + 7, 'x40.6', fontsize=8, color='red', ha='center', va='center')
ax1.text(3 - barwidth/2 - 0.05, nnd[3] + 25, 'x311.8', fontsize=8, color='red', ha='center', va='center')
ax1.text(4 - barwidth/2, nnd[2], '5\nclusters', fontsize=8, color='black', ha='center', va='center')
ax1.text(4 + barwidth/2, 8, 'OOM', fontsize=8, color=color_two, ha='center', va='center')
ax1.set_xlim((-0.5, 4.5))
ax1.tick_params(axis='y', labelcolor="green")

plot_color="#4b0082"
ax1.plot([0], [0], color=plot_color, label="trustworthiness scores")
ax1.legend(fontsize=8)

ax1.set_ylabel("Time (s)", color="green")


ax2 = ax1.twinx()
ax2.set_ylim((0.7, 1))
ax2.set_ylabel("Trustworthiness Score", color=plot_color)
ax2.tick_params(axis='y', labelcolor=plot_color)
ax2.set_yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
ax2.plot([0-barwidth/1.5, 0 + barwidth/1.5], [0.7807, 0.7832], marker='o', markersize=5, color=plot_color)
ax2.plot([1-barwidth/1.5, 1 + barwidth/1.5], [0.9236, 0.9149], marker='o', markersize=5, color=plot_color)
ax2.plot([2-barwidth/1.5, 2 + barwidth/1.5], [0.8929, 0.8829], marker='o', markersize=5, color=plot_color)
ax2.plot([3-barwidth/1.5, 3 + barwidth/1.5], [0.9156, 0.9030], marker='o', markersize=5, color=plot_color)
ax2.plot([4-barwidth/2], [0.8122], marker='o', markersize=5, color=plot_color)

plt.title("Time to run UMAP in RAPIDS cuML\nusing NN Descent VS Brute Force KNN (log scale)")
plt.savefig("./umap_result_tmp.png")
quit()



nnd = np.array([9933, 34030, 53411, 122979, 649325])/1000
bfk = np.array([214466, 2191373, 2170869, 38350718])/1000

fig, ax1 = plt.subplots()
ax1.set_yscale("log")

ax1.bar(x - barwidth/2, nnd, width=barwidth, color=color_one, label="Time to run UMAP with NN Descent")
ax1.bar(x[:4] + barwidth/2, bfk, width=barwidth, color=color_two, label="Time to run UMAP with Brute Force")
ax1.set_xticks(ticks=x)
ax1.set_xticklabels(["Gist\n(1M, 960)", "Amazon food\nreview\n(5M,384)","Deep Image\n(10M, 96)", "Amazon Electronics\nreview\n(20M, 384)", "Wiki-all\nsubsample\n(50M, 768)"], fontsize=8)
ax1.text(0 - barwidth/2, nnd[0] + 1, 'x21.6', fontsize=8, color='red', ha='center', va='center')
ax1.text(1 - barwidth/2, nnd[1] + 5, 'x64.4', fontsize=8, color='red', ha='center', va='center')
ax1.text(2 - barwidth/2, nnd[2] + 7, 'x40.6', fontsize=8, color='red', ha='center', va='center')
ax1.text(3 - barwidth/2 - 0.05, nnd[3] + 25, 'x311.8', fontsize=8, color='red', ha='center', va='center')
# ax1.text(4 - barwidth/2, nnd[2], '10\nclusters', fontsize=8, color='black', ha='center', va='center')
ax1.text(4 + barwidth/2, 8, 'OOM', fontsize=8, color=color_two, ha='center', va='center')
ax1.set_xlim((-0.5, 4.5))
ax1.tick_params(axis='y', labelcolor="green")

plot_color="#4b0082"
# ax1.plot([0], [0], color=plot_color, label="trustworthiness scores")
ax1.legend(fontsize=8)

ax1.set_ylabel("Time (s)", color="green")

# plot_color="#0071C5"

# ax2 = ax1.twinx()
# ax2.set_ylim((0.7, 1))
# ax2.set_ylabel("Trustworthiness Score", color=plot_color)
# ax2.tick_params(axis='y', labelcolor=plot_color)
# ax2.set_yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
# ax2.plot([0-barwidth/1.5, 0 + barwidth/1.5], [0.7807, 0.7832], marker='o', markersize=5, color=plot_color)
# ax2.plot([1-barwidth/1.5, 1 + barwidth/1.5], [0.9236, 0.9149], marker='o', markersize=5, color=plot_color)
# ax2.plot([2-barwidth/1.5, 2 + barwidth/1.5], [0.8929, 0.8829], marker='o', markersize=5, color=plot_color)
# ax2.plot([3-barwidth/1.5, 3 + barwidth/1.5], [0.9156, 0.9030], marker='o', markersize=5, color=plot_color)

# plt.ylabel("Time (s)")
plt.title("Time to run UMAP in RAPIDS cuML\nusing NN Descent VS Brute Force KNN (log scale)")
plt.savefig("./umap_result5.png")
quit()

x = np.array([0,1,2,3,4])
nnd = np.array([6568, 27265, 47077, 218233, 1394599])/1000
bfk = np.array([214095, 2146130, 2190538, 90636497])/1000

fig, ax1 = plt.subplots()
ax1.set_yscale("log")

ax1.bar(x - barwidth/2, nnd, width=barwidth, color=color_one, label="Time to build with NN Descent")
ax1.bar(x[:4] + barwidth/2, bfk, width=barwidth, color=color_two, label="Time to build with Brute Force")
ax1.set_xticks(ticks=x)
ax1.set_xticklabels(["Gist\n(1M, 960)", "Amazon food\nreview\n(5M,384)","Deep Image\n(10M, 96)", "Amazon clothes\nreview\n(32M, 384)", "Wiki-all\nsubsample\n(50M, 768)"], fontsize=8)
ax1.text(0 - barwidth/2, nnd[0] + 1, 'x32.6', fontsize=8, color='red', ha='center', va='center')
ax1.text(1 - barwidth/2, nnd[1] + 5, 'x78.7', fontsize=8, color='red', ha='center', va='center')
ax1.text(2 - barwidth/2, nnd[2] + 7, 'x46.5', fontsize=8, color='red', ha='center', va='center')
ax1.text(3 - barwidth/2, nnd[3] + 25, 'x415.3', fontsize=8, color='red', ha='center', va='center')
ax1.text(4 - barwidth/2, nnd[2], '10\nclusters', fontsize=8, color='black', ha='center', va='center')
ax1.text(4 + barwidth/2, 5, 'OOM', fontsize=8, color=color_two, ha='center', va='center')
ax1.set_xlim((-0.5, 4.5))
ax1.tick_params(axis='y', labelcolor="green")

plot_color="#4b0082"
ax1.plot([0], [0], color=plot_color, label="trustworthiness scores")
ax1.legend(fontsize=8)

ax1.set_ylabel("Time (s)", color="green")

# plot_color="#0071C5"

ax2 = ax1.twinx()
ax2.set_ylim((0.7, 1))
ax2.set_ylabel("Trustworthiness Score", color=plot_color)
ax2.tick_params(axis='y', labelcolor=plot_color)
ax2.set_yticks([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
ax2.plot([0-barwidth/1.5, 0 + barwidth/1.5], [0.7835, 0.7802], marker='o', markersize=5, color=plot_color)
ax2.plot([1-barwidth/1.5, 1 + barwidth/1.5], [0.9262, 0.9174], marker='o', markersize=5, color=plot_color)
ax2.plot([2-barwidth/1.5, 2 + barwidth/1.5], [0.8938, 0.8880], marker='o', markersize=5, color=plot_color)
ax2.plot([3-barwidth/1.5, 3 + barwidth/1.5], [0.8794, 0.8821], marker='o', markersize=5, color=plot_color)
ax2.plot([4-barwidth/2], [0.8118], marker='o', markersize=5, color=plot_color)

# plt.ylabel("Time (s)")
plt.title("Time to build knn graph in RAPIDS cuML\nusing NN Descent VS Brute Force KNN (log scale)")
plt.savefig("./umap_result3.png")
quit()
# x = [0,1,2,3,4]
# nnd = [6568, 27265, 47077, 218233, 1394599]
# # # x2 = [0,1,2,3]
# # # bfk = [214095, 2146130, 2190538, 2190538]
# # fig = plt.figure(figsize=(5, 5))
# # bax = brokenaxes(ylims=((0, 60000), (200000, 230000), (300000, 40000)), hspace=0.05)
# # bax.plot(x, nnd)

# # fig = plt.figure(figsize=(5, 2))
# bax = brokenaxes(ylims=((0, 60000), (200000, 230000)), hspace=.2)
# # x = np.linspace(0, 1, 100)
# x = np.array([0,1,2,3,4])
# bax.plot(x, nnd, label='sin')
# # bax.legend(loc=3)
# bax.set_xlabel('time')
# bax.set_ylabel('value')

# # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# # fig.subplots_adjust(hspace=0.05)
# # ax1.plot(x, nnd)
# # ax1.plot(x2, bfk)
# # ax2.plot(x, nnd)
# # ax2.plot(x2, bfk)
# # ax3.plot(x, nnd)
# # ax3.plot(x2, bfk)

# # ax1.set_ylim(300000, 40000) 
# # ax2.set_ylim(200000, 230000)  # outliers only
# # ax3.set_ylim(0, 60000)  # most of the data

# # ax1.spines.bottom.set_visible(False)
# # ax2.spines.top.set_visible(False)
# # ax2.spines.bottom.set_visible(False)
# # ax3.spines.top.set_visible(False)
# # # ax1.xaxis.tick_top()
# # ax1.tick_params(labeltop=False, labelbottom=False)  # don't put tick labels at the top
# # ax2.tick_params(labeltop=False)  # don't put tick labels at the top
# # ax3.tick_params(labeltop=False)  # don't put tick labels at the top
# # ax3.xaxis.tick_bottom()

# # d = .5  # proportion of vertical to horizontal extent of the slanted line
# # kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
# #               linestyle="none", color='k', mec='k', mew=1, clip_on=False)
# # ax1.plot([0, 230000], [0, 0], transform=ax1.transAxes, **kwargs)
# # ax2.plot([0, 230000], [230000, 230000], transform=ax2.transAxes, **kwargs)


# plt.savefig("./umap_result2.png")

# quit()



fig, axs = plt.subplots(1, 5, figsize=(20, 5))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
fig.suptitle('Time to build knn graph in RAPIDS cuML using NN Descent VS Brute Force KNN', fontsize=20)

width = 0.3
first = 0
second = 1
color_one = "#76B900"
color_two = "#008564"
axs[0].bar([first], [6568], color=color_one, label="NN Descent")
axs[0].bar([second], [214095], color=color_two, label="Brute Force")

yticks = np.array([50, 100, 150, 200])
axs[0].set_yticks(yticks * 1000)
axs[0].set_yticklabels([f"{i}" for i in yticks], fontsize=14)
axs[0].set_ylabel("Time (s)", fontsize=18)
axs[0].set_xticks([(first + second) / 2])
axs[0].set_xticklabels(["Gist (1M, 960)"], fontsize=14)
axs[0].text(0, 11000, 'x32.6', fontsize=14, color='red', ha='center', va='center')
axs[0].legend(loc='upper left')


axs[1].bar([first], [27265], color=color_one, label="NN Descent")
axs[1].bar([second], [2146130], color=color_two, label="Brute Force")

yticks = np.array([500, 1000, 1500, 2000])
axs[1].set_yticks(yticks * 1000)
axs[1].set_yticklabels([f"{i}" for i in yticks], fontsize=14)
axs[1].set_xticks([(first + second) / 2])
axs[1].set_xticklabels(["Amazon food review\n(5M, 384)"], fontsize=14)
axs[1].text(0, 80000, 'x78.7', fontsize=14, color='red', ha='center', va='center')


yticks = np.array([500, 1000, 1500, 2000])
axs[2].bar([first], [47077], color=color_one, label="NN Descent")
axs[2].bar([second], [2190538], color=color_two, label="Brute Force")
axs[2].set_yticks(yticks * 1000)
axs[2].set_yticklabels([f"{i}" for i in yticks], fontsize=14)
axs[2].set_xticks([(first + second) / 2])
axs[2].set_xticklabels(["Deep Image\n(10M, 96)"], fontsize=14)
axs[2].text(0, 100000, 'x46.5', fontsize=14, color='red', ha='center', va='center')

yticks = np.array([500, 1000, 1500, 2000])
axs[3].bar([first], [218233], color=color_one, label="NN Descent")
axs[3].bar([second], [0], color=color_two, label="Brute Force")
axs[3].set_yticks(yticks * 1000)
axs[3].set_yticklabels([f"{i}" for i in yticks], fontsize=14)
axs[3].set_xticks([(first + second) / 2])
axs[3].set_xticklabels(["Amazon clothes review\n(32M,384)"], fontsize=14)
axs[3].text(0, 300000, 'TBD', fontsize=14, color='red', ha='center', va='center')

yticks = np.array([500, 1000, 1500, 2000])
axs[4].bar([first], [1394599], color=color_one, label="NN Descent")
axs[4].bar([second], [0], color=color_two, label="Brute Force")
axs[4].set_yticks(yticks * 1000)
axs[4].set_yticklabels([f"{i}" for i in yticks], fontsize=14)
axs[4].set_xticks([(first + second) / 2])
axs[4].set_xticklabels(["Wiki-all subsample\n(50M, 768)"], fontsize=14)
axs[4].text(1, 100000, 'OOM', fontsize=14, color='red', ha='center', va='center')
axs[4].text(0, 500000, '10\nbatches', fontsize=14, color='black', ha='center', va='center')

plt.savefig("./umap_result.png")
