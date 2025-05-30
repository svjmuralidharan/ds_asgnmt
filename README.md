Community Detection Analysis
This project performs detailed analysis and comparison of network communities detected using multiple methods. It provides statistical summaries, similarity measures, and visualizations to help interpret and compare community structures.

📂 Project Structure

ds_asgnmt/
├── src/           # Main Python scripts (e.g., community_analysis_class.py)
├── test/          # Automated tests (pytest)
├── data/          # Place your input .parquet data files here (method_{number}.parquet)
├── output/        # Output CSVs, Markdown summary
├── figures/       # Plots and heatmaps
├── Dockerfile     # For containerized, reproducible runs
├── Makefile       # Project automation (build, run, test, clean)
├── requirements.txt # Python dependencies
└── README.md
🚀 Quick Start
1. Get the Data
Place your input .parquet files in the data/ folder, named as:

data/
├── method_1.parquet
├── method_2.parquet
├── method_3.parquet
├── method_4.parquet
└── method_5.parquet
2. Build and Run with Docker (Recommended)
All-in-one:

make all
Build Docker Image:

make build
Run the Analysis:

make run
Run Tests:

make test
Clean all outputs and summary files:

make clean
3. Where to Find the Outputs
output/: CSV files, Markdown summary (output.md)

figures/: PNG visualizations

Key files:

Community summaries: output/community_summary.csv

Similarity matrix: output/nmi_similarity_matrix.csv

Community alignment: output/aligned_communities_*.csv

Markdown summary: output/output.md

4. Running Locally (without Docker)
bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/community_analysis.py
PYTHONPATH=src pytest test/

📝 Notes
Adjust the number of input files as needed. The script auto-detects method_*.parquet files in data/.

Outputs and figures are always generated fresh with each run.

For advanced configuration, edit src/utils/constants.py.

Questions?
Open an issue or pull request!