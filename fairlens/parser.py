"""PDF resume parsing with PyMuPDF."""
try:
    import pymupdf as fitz
except ImportError:  # older PyMuPDF
    import fitz


class ParseError(Exception):
    pass


def pdf_to_text(data: bytes) -> str:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise ParseError(f"corrupt or unreadable PDF ({e.__class__.__name__})")
    text = "\n".join(page.get_text("text", sort=True) for page in doc)
    doc.close()
    if len(text.strip()) < 50:
        raise ParseError("no extractable text (scanned image PDF?)")
    return text
