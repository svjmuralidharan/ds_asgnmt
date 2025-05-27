import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
import os

# --- MAKE SURE FIGURE DIR EXISTS ---
fig_dir = "figure"
os.makedirs(fig_dir, exist_ok=True)

# 1. LOAD FILES
method_files = [f"data/method_{i}.parquet" for i in range(1, 6)]
method_names = [f"Method_{i}" for i in range(1, 6)]
community_assignments = {name: pd.read_parquet(file) for name, file in zip(method_names, method_files)}

# 2. EDA: COMMUNITY STATS
community_sizes = {}
summary_stats = []
for name, df in community_assignments.items():
    sizes = df.groupby("community")["node"].count()
    community_sizes[name] = sizes
    summary_stats.append(
        {
            "Method": name,
            "#Communities": len(sizes),
            "Largest": sizes.max(),
            "Smallest": sizes.min(),
            "Median Size": int(sizes.median()),
            "#Singletons": (sizes == 1).sum(),
        }
    )
summary_df = pd.DataFrame(summary_stats)
print("\nCommunity Assignment Summary:")
print(summary_df)

# 3. PLOT: COMMUNITY SIZE DISTRIBUTIONS
plt.figure(figsize=(18, 12))
for i, name in enumerate(method_names):
    plt.subplot(2, 3, i + 1)
    sizes = community_sizes[name].values
    plt.hist(sizes, bins=30, alpha=0.7)
    plt.title(f"{name}: Community Size Distribution")
    plt.xlabel("Community Size")
    plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "figure1_community_size_distributions.png"))
plt.close()

# 4. CALCULATE NMI/ARI MATRICES
all_nodes = sorted(set.union(*(set(df["node"]) for df in community_assignments.values())))
node_df = pd.DataFrame(index=all_nodes)
for name, df in community_assignments.items():
    node_map = dict(zip(df["node"], df["community"]))
    node_df[name] = [node_map.get(node, -1) for node in all_nodes]

N = len(method_names)
nmi_mat = np.zeros((N, N))
ari_mat = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        # Use string labels for both arrays
        labels1 = node_df[method_names[i]].astype(str).values
        labels2 = node_df[method_names[j]].astype(str).values

        # Combine unique labels for joint encoding
        all_labels = np.unique(np.concatenate([labels1, labels2]))
        label_map = {label: idx for idx, label in enumerate(all_labels)}
        labels1_codes = np.array([label_map[lbl] for lbl in labels1])
        labels2_codes = np.array([label_map[lbl] for lbl in labels2])

        nmi_mat[i, j] = normalized_mutual_info_score(labels1_codes, labels2_codes)
        ari_mat[i, j] = adjusted_rand_score(labels1_codes, labels2_codes)


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(nmi_mat, annot=True, xticklabels=method_names, yticklabels=method_names, cmap="YlGnBu")
plt.title("Normalized Mutual Information (NMI)")
plt.subplot(1, 2, 2)
sns.heatmap(ari_mat, annot=True, xticklabels=method_names, yticklabels=method_names, cmap="YlGnBu")
plt.title("Adjusted Rand Index (ARI)")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "figure2_NMI_ARI_heatmaps.png"))
plt.close()

# 5. IDENTIFY MOST SIMILAR PAIR
nmi_no_diag = nmi_mat.copy()
np.fill_diagonal(nmi_no_diag, -1)
best_i, best_j = np.unravel_index(np.argmax(nmi_no_diag), nmi_no_diag.shape)
method_A, method_B = method_names[best_i], method_names[best_j]
print(f"\nMost similar methods: {method_A} and {method_B} (NMI={nmi_mat[best_i, best_j]:.3f})")


# 6. ALIGN COMMUNITIES BETWEEN MOST SIMILAR PAIR
def get_community_map(df):
    comm_map = defaultdict(set)
    for _, row in df.iterrows():
        comm_map[row["community"]].add(row["node"])
    return comm_map


comm_A = get_community_map(community_assignments[method_A])
comm_B = get_community_map(community_assignments[method_B])

# Only compare pairs with shared nodes for efficiency
node_to_A = defaultdict(set)
node_to_B = defaultdict(set)
for comm, nodes in comm_A.items():
    for node in nodes:
        node_to_A[node].add(comm)
for comm, nodes in comm_B.items():
    for node in nodes:
        node_to_B[node].add(comm)
candidate_pairs = set()
for node in set(node_to_A) & set(node_to_B):
    for ca in node_to_A[node]:
        for cb in node_to_B[node]:
            candidate_pairs.add((ca, cb))

# Compute Jaccard similarity only for candidate pairs
similarities = []
for ca, cb in candidate_pairs:
    nodes_a = comm_A[ca]
    nodes_b = comm_B[cb]
    intersection = len(nodes_a & nodes_b)
    union = len(nodes_a | nodes_b)
    if union > 0:
        similarities.append((ca, cb, intersection / union))

# Greedy one-to-one matching
similarities.sort(key=lambda x: -x[2])
best_matches = {}
used_b = set()
for a, b, score in similarities:
    if a not in best_matches and b not in used_b:
        best_matches[a] = (b, score)
        used_b.add(b)
alignment_df = pd.DataFrame(
    [
        {"Community_" + method_A: a, "Community_" + method_B: b, "Jaccard_Similarity": score}
        for a, (b, score) in best_matches.items()
    ]
).sort_values(by="Jaccard_Similarity", ascending=False)

print(f"\nTop aligned communities between {method_A} and {method_B}:")
print(alignment_df.head(15).to_string(index=False))

# 7. PLOT: ALIGNED COMMUNITY HEATMAP (TOP 20)
N_heat = 20
top_matches = alignment_df.head(N_heat)
heatmap_data = pd.pivot_table(
    top_matches,
    values="Jaccard_Similarity",
    index="Community_" + method_A,
    columns="Community_" + method_B,
    fill_value=0,
)
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
plt.title(f"Top {N_heat} Community Matches: {method_A} vs {method_B}")
plt.xlabel(f"Communities in {method_B}")
plt.ylabel(f"Communities in {method_A}")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "figure3_aligned_community_heatmap.png"))
plt.close()