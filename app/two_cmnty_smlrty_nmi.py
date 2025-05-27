import pandas as pd
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import os

# --- SETUP ---
method_files = [f"data/method_{i}.parquet" for i in range(1, 6)]
method_names = [f"Method_{i}" for i in range(1, 6)]

out_dir = "output"

# --- LOAD DATA ---
community_assignments = {}
for name, file in zip(method_names, method_files):
    community_assignments[name] = pd.read_parquet(file)

# --- BUILD NODE x METHOD DATAFRAME ---
all_nodes = sorted(set.union(*(set(df['node']) for df in community_assignments.values())))
node_df = pd.DataFrame(index=all_nodes)
for name, df in community_assignments.items():
    node_map = dict(zip(df['node'], df['community']))
    node_df[name] = [node_map.get(node, -1) for node in all_nodes]

# --- COMPUTE NMI SIMILARITY MATRIX ---
N = len(method_names)
nmi_mat = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        labels1 = node_df[method_names[i]].astype(str).values
        labels2 = node_df[method_names[j]].astype(str).values
        all_labels = np.unique(np.concatenate([labels1, labels2]))
        label_map = {label: idx for idx, label in enumerate(all_labels)}
        labels1_codes = np.array([label_map[lbl] for lbl in labels1])
        labels2_codes = np.array([label_map[lbl] for lbl in labels2])
        nmi_mat[i, j] = normalized_mutual_info_score(labels1_codes, labels2_codes)

# --- FIND MOST SIMILAR PAIR (excluding self-similarity) ---
nmi_mat_no_diag = nmi_mat.copy()
np.fill_diagonal(nmi_mat_no_diag, -1)
best_i, best_j = np.unravel_index(np.argmax(nmi_mat_no_diag), nmi_mat_no_diag.shape)
method_A, method_B = method_names[best_i], method_names[best_j]
similarity_score = nmi_mat[best_i, best_j]

print(f"\nThe two most similar community detection methods are: {method_A} and {method_B}")
print(f"NMI similarity score: {similarity_score:.4f}")

# --- SAVE FULL NMI MATRIX AS CSV ---
nmi_df = pd.DataFrame(nmi_mat, index=method_names, columns=method_names)
nmi_df.to_csv(os.path.join(out_dir, "community_detection_nmi_similarity_matrix.csv"), index=False)
print("\nFull NMI similarity matrix saved to 'community_detection_nmi_similarity_matrix.csv'.")
