from ragbench.chunking import chunk_document, split_sections, word_windows

DOC = """Préambule du règlement.
Article 1
Objet du règlement.
Article 2
Champ d'application.
Article 6 bis
Disposition ajoutée."""


def test_split_sections_detects_articles():
    names = [name for name, _ in split_sections(DOC)]
    assert names == ["preambule", "Article 1", "Article 2", "Article 6 bis"]


def test_no_article_gives_single_section():
    assert split_sections("Texte libre sans structure.")[0][0] == "full"


def test_word_windows_overlap_and_coverage():
    words = [str(i) for i in range(23)]
    windows = list(word_windows(words, size=10, overlap=3))
    assert windows[0][-3:] == windows[1][:3]
    assert windows[-1][-1] == "22"
    assert all(len(w) <= 10 for w in windows)


def test_chunks_keep_section_keys():
    chunks = chunk_document("ai_act", DOC, max_words=5, overlap=1)
    keys = {c.section_key for c in chunks}
    assert "ai_act::Article 2" in keys
    assert len({c.id for c in chunks}) == len(chunks)
