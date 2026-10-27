from ragbench.chunking import Chunk
from ragbench.retrieval import BM25Retriever, reciprocal_rank_fusion, tokenize

CHUNKS_EN = [
    Chunk("c0", "gdpr", "Article 33", "Notification of a personal data breach to the supervisory authority"),
    Chunk("c1", "gdpr", "Article 17", "Right to erasure of personal data"),
    Chunk("c2", "ai_act", "Article 5", "Prohibited artificial intelligence practices"),
]

CHUNKS_FR = [
    Chunk("c0", "rgpd", "Article 33", "Notification d'une violation de données à l'autorité de contrôle"),
    Chunk("c1", "rgpd", "Article 17", "Droit à l'effacement des données à caractère personnel"),
    Chunk("c2", "ai_act", "Article 5", "Pratiques interdites en matière d'intelligence artificielle"),
]


def test_tokenize_english_stopwords():
    assert tokenize("The right to the erasure of data", "en") == ["right", "erasure", "data"]


def test_tokenize_french_strips_accents_and_stopwords():
    assert tokenize("Les données à l'effacement", "fr") == ["donnees", "effacement"]


def test_bm25_english():
    hits = BM25Retriever(CHUNKS_EN, "en").search("deadline to notify a data breach", k=2)
    assert hits[0][0] == 0


def test_bm25_french():
    hits = BM25Retriever(CHUNKS_FR, "fr").search("délai de notification d'une violation", k=2)
    assert hits[0][0] == 0


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[(1, 9.0), (2, 5.0)], [(2, 0.9), (1, 0.8)], [(2, 1.0)]])
    assert fused[0][0] == 2
