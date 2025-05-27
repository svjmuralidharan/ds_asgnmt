import pandas as pd
import numpy as np
import os

# --- Setup ---
method_files = [f"data/method_{i}.parquet" for i in range(1, 6)]
method_names = [f"Method_{i}" for i in range(1, 6)]
out_dir = "output"
os.makedirs(out_dir, exist_ok=True)

for name, file in zip(method_names, method_files):
    df = pd.read_parquet(file)
    comm_sizes = df.groupby('community')['node'].count().sort_values(ascending=False)
    findings = {}

    findings['#Communities'] = len(comm_sizes)
    findings['Largest Community Size'] = comm_sizes.max()
    findings['Smallest Community Size'] = comm_sizes.min()
    findings['Median Community Size'] = int(np.median(comm_sizes))
    findings['#Singleton Communities'] = (comm_sizes == 1).sum()
    findings['Largest Community ID'] = comm_sizes.idxmax()
    findings['Smallest Community ID'] = comm_sizes.idxmin()

    # Top 5 largest communities (ID and size)
    top5 = comm_sizes.head(5)
    for i, (comm_id, size) in enumerate(top5.items(), 1):
        findings[f'Top_{i}_Community_ID'] = comm_id
        findings[f'Top_{i}_Community_Size'] = size

    # Transpose: make one column for "Finding" and one for "Value"
    df_out = pd.DataFrame(findings, index=[0]).T.reset_index()
    df_out.columns = ['Finding', 'Value']
    df_out.to_csv(os.path.join(out_dir, f"{name}_interesting_findings.csv"), index=False)

print(f"Transposed interesting findings CSVs saved in '{out_dir}' directory.")
