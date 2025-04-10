from typing import List
from src.rag.components.data_sources.base import Document
from src.rag.components.chunking.base import BaseChunker
from src.rag.components.data_sources.text_splitter import RecursiveCharacterTextSplitter

class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators=None):
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)
