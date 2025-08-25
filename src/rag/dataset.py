import json
import jsonlines
from typing import List, Dict, Any
from pathlib import Path
import logging
from document_grouper import apply_grouping_if_enabled

logger = logging.getLogger(__name__)


class DSGVODataset:
    """DSGVO dataset preparation and management"""

    def __init__(self, data_path: str, config: Dict[str, Any] = None): # config ergänzt
        self.data_path = Path(data_path)
        self.config = config or {}  # neu hinzugefügt für Dokumentengruppierung
        self.documents: List[Dict[str, Any]] = []
        self.qa_pairs: List[Dict[str, Any]] = []

        # Check if evaluation dataset exists, otherwise load raw data
        #evaluation_path = Path("data/evaluation/dsgvo_evaluation_dataset.jsonl")
        #evaluation_path = Path("data/evaluation/dsgvo_llm_quality_dataset.jsonl")

        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        evaluation_path = PROJECT_ROOT / "data" / "evaluation" / "dsgvo_llm_quality_dataset.jsonl"

        if evaluation_path.exists():
            logger.info("Loading pre-prepared evaluation dataset")
            self._load_evaluation_dataset(evaluation_path)
        else:
            logger.info("Loading raw DSGVO documents and generating QA pairs")
            self._load_documents() # ← Lädt rohe Dokumente
            self._generate_qa_pairs() # ← Generiert QA-Paare automatisch

    def _load_evaluation_dataset(self, evaluation_path: Path):
        """Load pre-prepared evaluation dataset"""
        try:
            with jsonlines.open(evaluation_path, 'r') as reader:
                for obj in reader:
                    self.qa_pairs.append(obj)

            # Also load documents for indexing
            self._load_documents()

            logger.info(f"Loaded {len(self.qa_pairs)} pre-prepared QA pairs")

        except Exception as e:
            logger.error(f"Failed to load evaluation dataset: {e}")
            # Fallback to raw data
            self._load_documents()
            self._generate_qa_pairs()

    def _load_documents(self):
        """Load DSGVO documents from JSONL file"""
        try:
            with jsonlines.open(self.data_path, 'r') as reader:
                for obj in reader:
                    # Convert to standard document format
                    document = {
                        'id': obj.get('id', 'unknown'),
                        'text': obj.get('Text', ''),
                        'metadata': {
                            'kapitel_nr': obj.get('Kapitel_Nr', 0),
                            'kapitel_name': obj.get('Kapitel_Name', ''),
                            'artikel_nr': obj.get('Artikel_nr', 0),
                            'artikel_name': obj.get('Artikel_Name', ''),
                            'absatz_nr': obj.get('Absatz_nr', 0),
                            'unterabsatz_nr': obj.get('Unterabsatz_nr', 0),
                            'satz_nr': obj.get('Satz_nr', 0)
                        }
                    }
                    self.documents.append(document)

            logger.info(f"Loaded {len(self.documents)} DSGVO documents")

            self.documents = apply_grouping_if_enabled(self.documents, self.config) # neu hinzugefügt

        except Exception as e:
            logger.error(f"Failed to load DSGVO documents: {e}")
            raise

    def _generate_qa_pairs(self):
        """Generate question-answer pairs for evaluation (fallback only)"""
        # This is a simplified approach - in practice you might want more sophisticated QA generation
        qa_templates = [
            {
                'question': 'Was ist der Hauptzweck von DSGVO Artikel {artikel_nr}?',
                'context_field': 'artikel_name',
                'answer_field': 'text'
            },
            {
                'question': 'Was sagt DSGVO Artikel {artikel_nr} über {topic}?',
                'context_field': 'artikel_name',
                'answer_field': 'text',
                'topic_extraction': True
            },
            {
                'question': 'Was sind die wichtigsten Anforderungen in DSGVO Kapitel {kapitel_nr}?',
                'context_field': 'kapitel_name',
                'answer_field': 'text'
            }
        ]

        for doc in self.documents:
            metadata = doc['metadata']

            # Generate QA pairs based on templates
            for template in qa_templates:
                question = template['question'].format(
                    artikel_nr=metadata['artikel_nr'],
                    kapitel_nr=metadata['kapitel_nr'],
                    topic='data processing'  # Placeholder wird mit einem Standard-Wert gefüllt
                )

                # For topic extraction, create more specific questions
                if template.get('topic_extraction'):
                    # Extract key topics from the text
                    topics = self._extract_topics(doc['text'])
                    for topic in topics[:2]:  # Limit to 2 topics per document
                        specific_question = f"Was sagt DSGVO Artikel {metadata['artikel_nr']} über {topic}?"
                        qa_pair = {
                            'id': f"{doc['id']}_qa_{len(self.qa_pairs)}",
                            'question': specific_question,
                            'ground_truth': doc['text'],
                            'document_id': doc['id'],
                            'metadata': metadata
                        }
                        self.qa_pairs.append(qa_pair)
                else:
                    qa_pair = {
                        'id': f"{doc['id']}_qa_{len(self.qa_pairs)}",
                        'question': question,
                        'ground_truth': doc['text'],
                        'document_id': doc['id'],
                        'metadata': metadata
                    }
                    self.qa_pairs.append(qa_pair)

        logger.info(f"Generated {len(self.qa_pairs)} QA pairs")

    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text (simplified)"""
        # Simple keyword extraction - in practice you might use NLP libraries
        keywords = [
            'personenbezogene Daten', 'Verarbeitung', 'Einwilligung', 'Rechte',
            'Verantwortlicher', 'Auftragsverarbeiter', 'Aufsichtsbehörde',
            'Datenschutz', 'Grundrechte', 'Freiheiten', 'Schutz', 'Sicherheit'
        ]

        found_topics = []
        for keyword in keywords:
            if keyword.lower() in text.lower():
                found_topics.append(keyword)

        return found_topics[:3]  # Return up to 3 topics

    def get_documents(self) -> List[Dict[str, Any]]:
        """Get all documents"""
        return self.documents

    def get_qa_pairs(self) -> List[Dict[str, Any]]:
        """Get all QA pairs"""
        return self.qa_pairs

    def get_evaluation_subset(self, num_qa: int = 50) -> List[Dict[str, Any]]:
        """Get a subset of QA pairs for evaluation"""
        import random
        if num_qa >= len(self.qa_pairs):
            return self.qa_pairs

        return random.sample(self.qa_pairs, num_qa)

    def save_qa_pairs(self, output_path: str):
        """Save QA pairs to JSONL file"""
        try:
            with jsonlines.open(output_path, 'w') as writer:
                for qa_pair in self.qa_pairs:
                    writer.write(qa_pair)

            logger.info(f"Saved {len(self.qa_pairs)} QA pairs to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save QA pairs: {e}")
            raise
