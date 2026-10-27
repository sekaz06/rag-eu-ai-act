FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[dense,app]"

COPY app ./app
COPY data/processed ./data/processed

ENV LLM_BASE_URL=http://host.docker.internal:11434/v1
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
