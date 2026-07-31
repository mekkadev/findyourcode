from findyourcode.chunker import Chunk
from findyourcode.enrich import build_embed_text, path_words, split_identifiers


def test_identifier_splitting():
    words = split_identifiers("checkUserCredentials HTTPServerError issue_session_token")
    assert "check" in words and "user" in words and "credentials" in words
    assert "http" in words and "server" in words
    assert "session" in words and "token" in words


def test_stopwords_and_short_tokens_dropped():
    words = split_identifiers("self.return_value = a + bb")
    assert "self" not in words and "return" not in words
    assert "bb" not in words


def test_path_words():
    assert set(path_words("src/api/auth-flow/session_store.py")) >= {
        "src",
        "api",
        "auth",
        "flow",
        "session",
        "store",
    }


def test_embed_text_carries_context():
    chunk = Chunk(
        rel="src/api/session.py",
        lang="python",
        kind="method",
        symbol="issueTicket",
        parent="CredentialChecker",
        start_line=10,
        end_line=20,
        code="def issueTicket(self, login):\n    return sign(login)",
        doc="Creates a signed ticket for a signed-in client.",
    )
    text = build_embed_text(chunk)
    assert text.startswith("python method CredentialChecker.issueTicket")
    assert "file: src/api/session.py" in text
    assert "signed ticket" in text
    assert "issue" in text and "ticket" in text
    assert text.endswith("return sign(login)")
    assert text.index("about:") < text.index("names:") < text.index("file:")
