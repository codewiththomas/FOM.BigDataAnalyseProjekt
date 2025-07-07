from .base_chunker import BaseChunker
from typing import List

class LineChunker(BaseChunker):
    def __init__(self, config: dict = None):
        pass

    def split_document(self, document: str, chunk_size: int = None) -> List[str]:
        """Teilt Dokument zeilenweise auf"""
        lines = document.split('\n')
        chunks = []
        current_chunk = ""

        for line in lines:
            if len(current_chunk) + len(line) > (chunk_size or 1000):
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line
            else:
                current_chunk += "\n" + line if current_chunk else line

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks