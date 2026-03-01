"""
Extract plain text from any supported file format.
Supports: .txt, .md, .pdf, .docx, .xlsx, .csv
"""
import logging
import io
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text(file_path_or_bytes: str | io.BytesIO, ext: str = None) -> str:
    """
    Extract plain text from any supported file format.
    Supports file paths (str) or memory buffers (io.BytesIO).
    If using BytesIO, you must provide the extension (e.g. '.pdf').
    """
    try:
        if isinstance(file_path_or_bytes, str):
            ext = Path(file_path_or_bytes).suffix.lower()
            source = file_path_or_bytes
        else:
            ext = ext.lower() if ext else ""
            source = file_path_or_bytes

        if ext in (".txt", ".md"):
            if isinstance(source, str):
                return Path(source).read_text(encoding="utf-8", errors="ignore")
            else:
                return source.getvalue().decode("utf-8", errors="ignore")

        elif ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(source)
            return "\n\n".join(
                page.extract_text() or "" for page in reader.pages
            )

        elif ext == ".docx":
            from docx import Document
            doc = Document(source)
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

        elif ext in (".xlsx", ".xls"):
            import pandas as pd
            dfs = pd.read_excel(source, sheet_name=None)
            parts = []
            for sheet_name, df in dfs.items():
                parts.append(f"[Sheet: {sheet_name}]\n{df.to_string(index=False)}")
            return "\n\n".join(parts)

        elif ext == ".csv":
            import pandas as pd
            df = pd.read_csv(source)
            return df.to_string(index=False)

        else:
            source_name = source if isinstance(source, str) else "BytesIO"
            logger.warning(f"Unsupported file type: {ext} for {source_name}")
            return ""

    except Exception as e:
        source_name = file_path_or_bytes if isinstance(file_path_or_bytes, str) else "BytesIO"
        logger.error(f"Text extraction failed for {source_name}: {e}")
        return ""
