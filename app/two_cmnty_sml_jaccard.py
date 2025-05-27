import pandas as pd
import numpy as np
from collections import defaultdict
import os

# --- SETUP ---
method_files = [f"data/method_{i}.parquet" for i in range(1, 6)]
method_names = [f"Method_{i}" for i in range(1, 6)]

out_dir = "output"

# --- LOAD DATA ---
community_assignments = {}
for name, file in zip(method_names, method_files):
    community_assignments[name] = pd.read_parquet(file)

# --- FUNCTION TO BUILD community->set(nodes) MAP ---
def get_community_map(df):
    comm_map = defaultdict(set)
    for _, row in df.iterrows():
        comm_map[row["community"]].add(row["node"])
    return comm_map

# --- CALCULATE PAIRWISE JACCARD SIMILARITIES ---
def average_best_jaccard(comm_map_A, comm_map_B):
    scores = []
    for comm_a, nodes_a in comm_map_A.items():
        best = 0
        for comm_b, nodes_b in comm_map_B.items():
            intersection = len(nodes_a & nodes_b)
            union = len(nodes_a | nodes_b)
            if union > 0:
                jaccard = intersection / union
                if jaccard > best:
                    best = jaccard
        scores.append(best)
    return np.mean(scores) if scores else 0

N = len(method_names)
jaccard_mat = np.zeros((N, N))

comm_maps = [get_community_map(community_assignments[name]) for name in method_names]

for i in range(N):
    for j in range(N):
        if i == j:
            jaccard_mat[i, j] = 1.0  # perfect match to self
        else:
            jaccard_mat[i, j] = average_best_jaccard(comm_maps[i], comm_maps[j])

# --- FIND MOST SIMILAR PAIR (excluding self) ---
jaccard_mat_no_diag = jaccard_mat.copy()
np.fill_diagonal(jaccard_mat_no_diag, -1)
best_i, best_j = np.unravel_index(np.argmax(jaccard_mat_no_diag), jaccard_mat_no_diag.shape)
method_A, method_B = method_names[best_i], method_names[best_j]
similarity_score = jaccard_mat[best_i, best_j]

print(f"\nThe two most similar community detection methods (by average best Jaccard) are: {method_A} and {method_B}")
print(f"Jaccard similarity score: {similarity_score:.4f}")

# --- SAVE FULL JACCARD MATRIX AS CSV ---
jaccard_df = pd.DataFrame(jaccard_mat, index=method_names, columns=method_names)
jaccard_df.to_csv(os.path.join(out_dir, "community_detection_jaccard_similarity_matrix.csv"), index=False)
print("\nFull Jaccard similarity matrix saved to 'community_detection_jaccard_similarity_matrix.csv'.")
