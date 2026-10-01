from app.ingest import chunk_text


def test_long_text_makes_multiple_chunks():
    assert len(chunk_text("word " * 500, chunk_size=200, overlap=20)) > 1


def test_chunks_respect_max_size():
    for c in chunk_text("abcdefghij " * 300, chunk_size=200, overlap=20):
        assert len(c) <= 200


def test_overlap_applied():
    text = "".join(chr(97 + i % 26) for i in range(500))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert chunks[0][-20:] == chunks[1][:20]


def test_empty_and_whitespace_return_nothing():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_very_short_text_ignored():
    assert chunk_text("hi") == []


def test_short_text_single_chunk():
    text = "This is a short but valid sentence for one chunk."
    assert chunk_text(text, chunk_size=1000, overlap=150) == [text]


def test_whitespace_normalized():
    assert "  " not in chunk_text("a  b\n\nc " * 10, chunk_size=1000, overlap=10)[0]
