import re
import io
from typing import List, Dict, Any

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None


class TextProcessor:
    """Handles document extraction (PDF, DOCX, TXT) and text preprocessing/sentence segmentation."""

    @staticmethod
    def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
        """Extract plain text from uploaded file bytes according to file type extension."""
        ext = filename.split(".")[-1].lower() if "." in filename else ""
        
        if ext == "pdf":
            return TextProcessor._extract_pdf(file_bytes)
        elif ext in ["docx", "doc"]:
            return TextProcessor._extract_docx(file_bytes)
        elif ext in ["txt", "md", "csv", "json"]:
            return file_bytes.decode("utf-8", errors="ignore")
        else:
            # Fallback to UTF-8 decoding
            return file_bytes.decode("utf-8", errors="ignore")

    @staticmethod
    def _extract_pdf(file_bytes: bytes) -> str:
        if pypdf is None:
            raise RuntimeError("pypdf is not installed.")
        pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        extracted_pages = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_pages.append(text)
        return "\n\n".join(extracted_pages)

    @staticmethod
    def _extract_docx(file_bytes: bytes) -> str:
        if docx is None:
            raise RuntimeError("python-docx is not installed.")
        doc = docx.Document(io.BytesIO(file_bytes))
        full_text = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n\n".join(full_text)

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean document text (remove headers/footers, excess whitespace)."""
        if not text:
            return ""
        # Remove unusual unicode control chars
        text = re.sub(r'[\r\t]', ' ', text)
        # Normalize multiple spaces/newlines
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()

    @staticmethod
    def segment_into_sentences(text: str) -> List[str]:
        """Break document into discrete sentences for fine-grained semantic similarity & heatmap alignment."""
        cleaned = TextProcessor.clean_text(text)
        if not cleaned:
            return []

        # Sentence splitting pattern supporting punctuation followed by space or newline
        sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+')
        raw_sentences = sentence_endings.split(cleaned)
        
        sentences = []
        for s in raw_sentences:
            s_clean = s.strip()
            # Filter out trivially tiny non-word sentences
            if len(s_clean) > 5 and re.search(r'\w', s_clean):
                sentences.append(s_clean)

        # Fallback if text couldn't be split into sentences
        if not sentences and cleaned:
            sentences = [cleaned]

        return sentences

    @staticmethod
    def segment_into_paragraphs(text: str) -> List[str]:
        """Break document into paragraphs."""
        cleaned = TextProcessor.clean_text(text)
        paragraphs = [p.strip() for p in cleaned.split('\n\n') if p.strip()]
        return paragraphs if paragraphs else ([cleaned] if cleaned else [])
