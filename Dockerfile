FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything in the src folder into /app/src in the container
COPY src/ ./src/
# copy your data directory in the container
COPY data/ ./data/
# copy your test directory in the container
COPY test/ ./test/

# Run Script
CMD ["python", "src/community_analysis.py"]
