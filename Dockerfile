FROM python:3.11-slim

WORKDIR /app

# System deps for lxml / scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libxml2-dev libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
# docs/ created at runtime for graph diagrams
RUN mkdir -p docs

ENV PYTHONPATH=/app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
