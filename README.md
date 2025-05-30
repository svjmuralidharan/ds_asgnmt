# Community Detection Analysis

This project performs detailed analysis and comparison of network communities detected using five different methods. It provides statistical summaries, similarity measures, and visualizations to help interpret and compare community structures.

---

## 📂 Project Structure

ds_asgnmt/
├── src/ # Main Python scripts (e.g., community_analysis_class.py)
├── test/ # Automated tests (pytest)
├── data/ # Place your input .parquet data files here in the format method_{number}.parquet (Maximum 5 files)
├── output/ # Output CSVs, Markdown summary
├── figures/ # Plots and heatmaps
├── Dockerfile # For containerized, reproducible runs
├── Makefile # Project automation (build, run, test, clean)
├── requirements.txt # Python dependencies
└── README.md


---

## 🚀 Quick Start

### 1. **Get the Data**

Place your input `.parquet` files in the `data/` folder:

data/
├── method_1.parquet
├── method_2.parquet
├── method_3.parquet
├── method_4.parquet
└── method_5.parquet


### 2. **Build and Run with Docker (Recommended)**

Build the Docker image (first time only):

```bash
make all -- To build the Docker Image and Run the program
make build -- To build Docker Image
make run -- To execute the Script
make test -- To execute the Tests
make clean -- To cleanup the output files and figures.

output/: CSV files and output.md Markdown summary
figures/: PNG visualizations of analysis

Running Locally (without Docker)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/community_analysis.py
PYTHONPATH=src pytest test/

Outputs
Community summaries: output/community_summary.csv

Similarity matrix: output/nmi_similarity_matrix.csv

Community alignment: output/aligned_communities_*.csv

Markdown summary: output/output.md

Figures: All PNG images in figures/

make test
# or, if running locally:
PYTHONPATH=src pytest test/

