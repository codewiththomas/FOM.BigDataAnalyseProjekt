"""
Implementiert Dokumentenlader für verschiedene Dateitypen.
"""

import os
from pathlib import Path
from typing import List, Optional

from .base import Document, DocumentLoader


class TextFileLoader(DocumentLoader):
    """
    Lädt Dokumente aus Textdateien.
    """

    def __init__(self, file_path: str, encoding: str = "utf-8"):
        """
        Initialisiert den TextFileLoader.

        Args:
            file_path: Pfad zur Textdatei
            encoding: Encoding der Textdatei (default: utf-8)
        """
        self.file_path = file_path
        self.encoding = encoding

    def load(self) -> List[Document]:
        """
        Lädt ein Dokument aus einer Textdatei.

        Returns:
            Eine Liste mit einem Document-Objekt
        """
        try:
            with open(self.file_path, "r", encoding=self.encoding) as file:
                content = file.read()

            metadata = {
                "source": self.file_path,
                "file_name": os.path.basename(self.file_path)
            }

            return [Document(content=content, metadata=metadata)]
        except Exception as e:
            raise ValueError(f"Fehler beim Laden der Datei {self.file_path}: {e}")


class DirectoryLoader(DocumentLoader):
    """
    Lädt Dokumente aus einem Verzeichnis.
    """

    def __init__(
        self,
        directory_path: str,
        glob_pattern: str = "*.txt",
        encoding: str = "utf-8"
    ):
        """
        Initialisiert den DirectoryLoader.

        Args:
            directory_path: Pfad zum Verzeichnis
            glob_pattern: Muster für die zu ladenden Dateien (default: *.txt)
            encoding: Encoding der Textdateien (default: utf-8)
        """
        self.directory_path = directory_path
        self.glob_pattern = glob_pattern
        self.encoding = encoding

    def load(self) -> List[Document]:
        """
        Lädt Dokumente aus einem Verzeichnis.

        Returns:
            Eine Liste von Document-Objekten
        """
        path = Path(self.directory_path)
        documents = []

        for file_path in path.glob(self.glob_pattern):
            loader = TextFileLoader(str(file_path), self.encoding)
            try:
                documents.extend(loader.load())
            except Exception as e:
                print(f"Warnung: Konnte Datei {file_path} nicht laden: {e}")

        return documents