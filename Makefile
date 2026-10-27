.PHONY: install ingest eval app test lint

install:
	pip install -e ".[all]"

ingest:
	python -m ragbench.ingest

eval:
	python -m ragbench.eval.run --configs bm25 dense hybrid hybrid_rerank --k 5

app:
	streamlit run app/streamlit_app.py

test:
	pytest -q

lint:
	ruff check .
