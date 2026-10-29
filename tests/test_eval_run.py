from ragbench.chunking import Chunk
from ragbench.eval.run import check_questions, unknown_labels

CHUNKS = [Chunk("c0", "gdpr", "Article 33", "text"), Chunk("c1", "ai_act", "Article 5", "text")]


def test_unknown_labels_flags_typos():
    qs = [{"id": "q1", "relevant": {"gdpr::Article 33"}}, {"id": "q2", "relevant": {"rgpd::Article 33"}}]
    assert unknown_labels(qs, {c.section_key for c in CHUNKS}) == {"q2": {"rgpd::Article 33"}}


def test_out_of_corpus_questions_are_skipped():
    qs = [
        {"id": "q1", "relevant": {"gdpr::Article 33"}, "verified": True},
        {"id": "o1", "relevant": set(), "verified": True},
    ]
    assert [q["id"] for q in check_questions(qs, CHUNKS)] == ["q1"]
