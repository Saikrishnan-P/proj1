FROM python:3.11-slim

WORKDIR /app

# git is needed for /ingest's repo_url clone path; build-essential covers
# native deps some of chromadb/sentence-transformers' own deps pull in.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the embedding model into the image so the first real request doesn't
# pay a ~80MB download + load penalty and risk failing Railway's health
# check during cold start.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

RUN mkdir -p /app/data/chroma

EXPOSE 8000

# $PORT is injected by Railway; defaults to 8000 for other hosts/local docker run.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
