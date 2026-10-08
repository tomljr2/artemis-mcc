from data.prepare_fineweb_edu import clean_document, split_documents


def test_curly_quotes_and_tabs_become_plain_ascii_like_the_books():
    assert clean_document("“Go” for\tlaunch – today’s plan\n") == '"Go" for launch - today\'s plan'


def test_paragraph_breaks_are_kept_but_long_runs_of_blank_lines_are_squeezed():
    assert clean_document("First.\n\n\n\nSecond.\nThird.") == "First.\n\nSecond.\nThird."


def test_documents_with_other_scripts_or_symbols_are_skipped():
    # Our NASA text is plain ASCII. Rather than cut letters out of words ("café" -> "caf"),
    # skip the whole document: there are plenty more.
    assert clean_document("A café near the launch site.") is None
    assert clean_document("Orbit: 2πr") is None


def test_every_hundredth_document_is_held_out_whole():
    docs = [f"doc {i}" for i in range(250)]
    train, val = split_documents(docs, val_every=100)
    assert val == ["doc 0", "doc 100", "doc 200"]
    assert len(train) == 247
    assert not set(train) & set(val)  # split by document: nothing in both
