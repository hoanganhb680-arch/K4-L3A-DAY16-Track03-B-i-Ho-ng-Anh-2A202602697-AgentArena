import unicodedata


def normalise(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def supports(text: str, body: str) -> bool:
    claim = normalise(text)
    return len(claim) >= 12 and any(claim in normalise(line) for line in body.splitlines())

def observed_sources(ctx):
    if ctx.corpus is None:
        return []
    if "observed_docs" in ctx.state:
        seen = ctx.state["observed_docs"]
        return [doc for doc in ctx.corpus.docs if doc.doc_id in seen]
    observed = ctx.observed_text
    return [doc for doc in ctx.corpus.docs if doc.body and doc.body in observed]
