#!/bin/zsh
source .venv/bin/activate
export EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
export RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
export LLM_MODEL=qwen2.5:3b-instruct
streamlit run app/streamlit_app.py