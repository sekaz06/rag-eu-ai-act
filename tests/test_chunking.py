from ragbench.chunking import chunk_document, split_sections, word_windows

DOC = """Préambule du règlement.
Article 1
Objet du règlement.
Article 2
Champ d'application.
Article 3
Disposition finale."""


def test_split_sections_detects_articles():
    names = [name for name, _ in split_sections(DOC)]
    assert names == ["preambule", "Article 1", "Article 2", "Article 3"]


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


def test_cross_reference_is_not_a_heading():
    text = "Article 5\nProhibited practices referred to in\nArticle 6(2) are covered.\nArticle 6\nHigh-risk."
    names = [name for name, _ in split_sections(text)]
    assert names == ["Article 5", "Article 6"]


def test_wrapped_cross_reference_is_ignored():
    # "Article 6 of Regulation ..." at a line start inside Article 5 must not open a section,
    # and the real "Article 6" heading alone on its line must be kept.
    text = (
        "Article 4\nDefinitions.\nArticle 5\nProhibited practices, see\n"
        "Article 6 of Regulation (EU) 2016/679.\nArticle 6\nHigh-risk systems.\n"
        "Article 2 of this Regulation applies.\nArticle 7\nAmendments."
    )
    names = [name for name, _ in split_sections(text)]
    assert names == ["Article 4", "Article 5", "Article 6", "Article 7"]
    sections = dict(split_sections(text))
    assert "Regulation (EU) 2016/679" in sections["Article 5"]
    assert "Article 2 of this Regulation" in sections["Article 6"]


def test_annexes_after_last_article():
    text = "Article 1\nScope.\nArticle 2\nEntry into force.\nANNEX I\nList one.\nANNEX III\nHigh-risk areas."
    names = [name for name, _ in split_sections(text)]
    assert names == ["Article 1", "Article 2", "Annex I", "Annex III"]
