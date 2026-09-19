from pathlib import Path
from typing import List

from langchain_core.documents import Document

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    BSHTMLLoader,
    Docx2txtLoader,
)

class DocumentLoader:
    """
    Loads documents of different formats and converts them into
    LangChain Document objects.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".txt",
        ".md",
        ".html",
        ".htm",
	".docx",
    }

    def load(self, file_path: str) -> List[Document]:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"{file_path} not found.")

        suffix = path.suffix.lower()

        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {suffix}"
            )

        if suffix == ".pdf":
            loader = PyMuPDFLoader(str(path))

        elif suffix in [".txt", ".md"]:
            loader = TextLoader(str(path), encoding="utf-8")

	elif suffix == ".docx":
    	    loader = Docx2txtLoader(str(path))

        else:
            loader = BSHTMLLoader(str(path))

        documents = loader.load()

        return documents