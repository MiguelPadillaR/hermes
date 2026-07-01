FROM python:3.13

# Set the working directory inside the container
WORKDIR /hermes_app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install essential system dependencies required for compiling rapidfuzz/chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install 'uv' globally inside the container to handle lightning-fast installations
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy your dependency files first (this leverages Docker caching)
COPY pyproject.toml uv.lock ./

# Install project dependencies globally inside the container using uv
RUN uv pip install --system -r pyproject.toml

# Copy the rest of your application code into the container
COPY . .

# Expose the default port that Streamlit listens on
EXPOSE 8501

# Define the command to launch your interactive Streamlit app
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]