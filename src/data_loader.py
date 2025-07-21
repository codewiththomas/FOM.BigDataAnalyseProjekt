"""
Data loader module for loading and chunking documents.
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import tiktoken  # type: ignore
from tqdm import tqdm  # type: ignore


class DocumentChunker:
    """Handles document chunking with fixed-size strategy."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, encoding_name: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))

    def chunk_text(self, text: str, source: str = "") -> List[Dict[str, Any]]:
        """
        Chunk text into overlapping segments.

        Args:
            text: Text to chunk
            source: Source document name

        Returns:
            List of chunk dictionaries with metadata
        """
        tokens = self.encoding.encode(text)
        chunks: List[Dict[str, Any]] = []

        start = 0
        chunk_id = 0

        while start < len(tokens):
            # Calculate end position
            end = min(start + self.chunk_size, len(tokens))

            # Extract chunk tokens and decode
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            # Create chunk metadata
            chunk = {
                "chunk_id": f"{source}_{chunk_id}" if source else str(chunk_id),
                "text": chunk_text,
                "source": source,
                "start_token": start,
                "end_token": end,
                "token_count": len(chunk_tokens)
            }

            chunks.append(chunk)

            # Move start position with overlap
            start = end - self.chunk_overlap
            chunk_id += 1

            # Break if we're at the end
            if end == len(tokens):
                break

        return chunks


class DataLoader:
    """Handles loading documents and test questions."""

    def __init__(self, documents_path: str, test_questions_path: str):
        self.documents_path = Path(documents_path)
        self.test_questions_path = Path(test_questions_path)
        self.chunker: Optional[DocumentChunker] = None

    def set_chunker(self, chunker: DocumentChunker):
        """Set the document chunker."""
        self.chunker = chunker

    def load_documents(self, file_extensions: List[str] = [".txt", ".md"]) -> List[Dict[str, Any]]:
        """
        Load all documents from the documents directory.

        Args:
            file_extensions: List of file extensions to load

        Returns:
            List of document dictionaries
        """
        documents: List[Dict[str, Any]] = []

        if not self.documents_path.exists():
            print(f"Documents path {self.documents_path} does not exist. Creating directory...")
            self.documents_path.mkdir(parents=True, exist_ok=True)
            return documents

        # Find all matching files
        files_to_load: List[Path] = []
        for ext in file_extensions:
            files_to_load.extend(self.documents_path.glob(f"*{ext}"))

        print(f"Found {len(files_to_load)} documents to load...")

        for file_path in tqdm(files_to_load, desc="Loading documents"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                document = {
                    "filename": file_path.name,
                    "path": str(file_path),
                    "content": content,
                    "size": len(content)
                }

                documents.append(document)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return documents

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk all documents using the configured chunker.

        Args:
            documents: List of document dictionaries

        Returns:
            List of chunk dictionaries
        """
        if not self.chunker:
            raise ValueError("No chunker set. Call set_chunker() first.")

        all_chunks: List[Dict[str, Any]] = []

        print(f"Chunking {len(documents)} documents...")

        for doc in tqdm(documents, desc="Chunking documents"):
            chunks = self.chunker.chunk_text(
                text=doc["content"],
                source=doc["filename"]
            )

            # Add document metadata to chunks
            for chunk in chunks:
                chunk["document_path"] = doc["path"]
                chunk["document_size"] = doc["size"]

            all_chunks.extend(chunks)

        print(f"Created {len(all_chunks)} chunks total")
        return all_chunks

    def load_test_questions(self) -> List[Dict[str, Any]]:
        """
        Load test questions from JSON file.

        Returns:
            List of test question dictionaries
        """
        if not self.test_questions_path.exists():
            print(f"Test questions file {self.test_questions_path} does not exist.")
            return []

        try:
            with open(self.test_questions_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)

            print(f"Loaded {len(questions)} test questions")
            return questions

        except Exception as e:
            print(f"Error loading test questions: {e}")
            return []

    def save_test_questions(self, questions: List[Dict[str, Any]]):
        """
        Save test questions to JSON file.

        Args:
            questions: List of test question dictionaries
        """
        # Create directory if it doesn't exist
        self.test_questions_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.test_questions_path, 'w', encoding='utf-8') as f:
                json.dump(questions, f, ensure_ascii=False, indent=2)

            print(f"Saved {len(questions)} test questions to {self.test_questions_path}")

        except Exception as e:
            print(f"Error saving test questions: {e}")


def create_sample_test_questions() -> List[Dict[str, Any]]:
    """Create sample test questions for DSGVO."""
    return [
        {
            "id": 1,
            "question": "Was ist die DSGVO und wann trat sie in Kraft?",
            "reference_answer": "Die DSGVO (Datenschutz-Grundverordnung) ist eine EU-Verordnung zum Schutz personenbezogener Daten, die am 25. Mai 2018 in Kraft getreten ist.",
            "category": "basics"
        },
        {
            "id": 2,
            "question": "Welche Rechte haben betroffene Personen nach der DSGVO?",
            "reference_answer": "Betroffene Personen haben unter anderem das Recht auf Auskunft, Berichtigung, Löschung, Einschränkung der Verarbeitung, Datenübertragbarkeit und Widerspruch.",
            "category": "rights"
        },
        {
            "id": 3,
            "question": "Was sind die Grundsätze für die Verarbeitung personenbezogener Daten?",
            "reference_answer": "Die Grundsätze umfassen Rechtmäßigkeit, Fairness, Transparenz, Zweckbindung, Datenminimierung, Richtigkeit, Speicherbegrenzung und Integrität/Vertraulichkeit.",
            "category": "principles"
        },
        {
            "id": 4,
            "question": "Wie hoch können Bußgelder nach der DSGVO sein?",
            "reference_answer": "Bußgelder können bis zu 20 Millionen Euro oder bis zu 4% des weltweiten Jahresumsatzes des Unternehmens betragen, je nachdem welcher Betrag höher ist.",
            "category": "penalties"
        },
        {
            "id": 5,
            "question": "Was ist eine Datenschutz-Folgenabschätzung und wann ist sie erforderlich?",
            "reference_answer": "Eine Datenschutz-Folgenabschätzung ist erforderlich, wenn eine Verarbeitung voraussichtlich ein hohes Risiko für die Rechte und Freiheiten natürlicher Personen zur Folge hat.",
            "category": "assessment"
        },
        {
            "id": 6,
            "question": "Welche Rechtsgrundlagen gibt es für die Datenverarbeitung?",
            "reference_answer": "Die DSGVO nennt sechs Rechtsgrundlagen: Einwilligung, Vertragserfüllung, rechtliche Verpflichtung, Schutz lebenswichtiger Interessen, öffentliches Interesse und berechtigte Interessen.",
            "category": "legal_basis"
        },
        {
            "id": 7,
            "question": "Was bedeutet Privacy by Design?",
            "reference_answer": "Privacy by Design bedeutet, dass Datenschutz bereits bei der Entwicklung von Systemen und Verfahren berücksichtigt wird.",
            "category": "principles"
        },
        {
            "id": 8,
            "question": "Wann muss ein Datenschutzbeauftragter benannt werden?",
            "reference_answer": "Ein Datenschutzbeauftragter muss unter anderem bei öffentlichen Stellen oder wenn die Kerntätigkeit in umfangreicher Überwachung oder Verarbeitung besonderer Datenkategorien besteht, benannt werden.",
            "category": "dpo"
        },
        {
            "id": 9,
            "question": "Was ist bei der internationalen Datenübertragung zu beachten?",
            "reference_answer": "Die Übertragung in Drittländer ist nur mit Angemessenheitsbeschluss, geeigneten Garantien oder in Ausnahmefällen zulässig.",
            "category": "transfer"
        },
        {
            "id": 10,
            "question": "Welche Fristen gibt es für die Meldung von Datenschutzverletzungen?",
            "reference_answer": "Datenschutzverletzungen müssen innerhalb von 72 Stunden nach Bekanntwerden an die Aufsichtsbehörde gemeldet werden. Bei hohem Risiko müssen auch die betroffenen Personen unverzüglich informiert werden.",
            "category": "breach"
        }
    ]


if __name__ == "__main__":
    # Example usage
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
    loader = DataLoader("../data/documents", "../data/test_questions.json")
    loader.set_chunker(chunker)

    # Create sample test questions
    sample_questions = create_sample_test_questions()
    loader.save_test_questions(sample_questions)

    print("Data loader setup complete!")