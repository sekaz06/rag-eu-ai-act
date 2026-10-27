"""Minimal demo UI: `streamlit run app/streamlit_app.py`."""

import streamlit as st

from ragbench.pipeline import CONFIGS, get_pipeline

st.set_page_config(page_title="EU AI Act & GDPR — grounded assistant", page_icon="⚖️", layout="wide")
st.title("⚖️ EU AI Act & GDPR Assistant")
st.caption("Hybrid RAG (BM25 + dense + reranking) — every answer cites its sources.")

with st.sidebar:
    config = st.selectbox("Retrieval configuration", CONFIGS, index=len(CONFIGS) - 1)
    k = st.slider("Number of passages", 1, 10, 5)

question = st.text_input("Your question", placeholder="Which AI practices are prohibited under the AI Act?")

if question:
    pipe = get_pipeline(config)
    with st.spinner("Retrieving and generating…"):
        answer = pipe.ask(question, k=k)
    st.markdown(answer.text)
    st.subheader("Sources")
    for i, c in enumerate(answer.sources, start=1):
        with st.expander(f"[{i}] {c.doc_id} — {c.section}"):
            st.write(c.text)
