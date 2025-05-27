FROM python:3.11

# Install Poetry
RUN pip install poetry

WORKDIR /app

# Copy only the dependency files first (for Docker layer cache)
COPY pyproject.toml poetry.lock* /app/

# Install dependencies (no virtualenv, since Docker is already isolated)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy the rest of your code
COPY . /app

# Set default command
CMD ["python", "your_script.py"]
