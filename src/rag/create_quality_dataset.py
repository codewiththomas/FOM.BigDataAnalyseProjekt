#!/usr/bin/env python3
"""
LLM-basierter DSGVO QA-Datensatz-Generator
Verwendet GPT-4o für intelligente Fragen-Generierung und Validierung
"""

import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from collections import defaultdict
import logging
import os
from openai import OpenAI
from dotenv import load_dotenv

# Lade .env Datei
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMQAGenerator:
    """Generiert qualitativ hochwertige QA-Paare mit GPT-4o"""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.documents = []
        self.articles = defaultdict(list)
        self.qa_pairs = []

    def load_documents(self, data_path: str):
        """Lädt und gruppiert DSGVO-Dokumente nach Artikeln"""
        logger.info("Lade DSGVO-Dokumente...")

        with jsonlines.open(data_path, 'r') as reader:
            for obj in reader:
                self.documents.append(obj)
                article_nr = obj.get('Artikel_nr', 0)
                self.articles[article_nr].append(obj)

        logger.info(f"Geladen: {len(self.documents)} Dokumente in {len(self.articles)} Artikeln")

    def get_article_context(self, article_nr: int) -> str:
        """Holt den vollständigen Kontext eines Artikels"""
        if article_nr not in self.articles:
            return ""

        article_texts = []
        for doc in self.articles[article_nr]:
            article_texts.append(doc.get('Text', ''))

        return " ".join(article_texts)


    def generate_question_with_llm(self, text: str, article_nr: int, artikel_name: str,
                                 article_context: str) -> Optional[Dict[str, Any]]:
        """Generiert eine intelligente Frage mit GPT-4o"""

        prompt = f"""
Du bist ein Experte für die DSGVO (Datenschutz-Grundverordnung).

Gegeben ist folgender Satz aus DSGVO Artikel {article_nr} ({artikel_name}):
"{text}"

Und hier ist der vollständige Kontext des Artikels:
"{article_context}"

Deine Aufgabe:
1. Analysiere den Satz und den Artikel-Kontext
2. Generiere eine NATÜRLICHE, KONKRETE Frage, die sich mit diesem Satz beantworten lässt
3. Bestimme den Fragetyp (rights, obligations, purpose, scope, definition, principles, general)
4. Bestimme die Schwierigkeit (easy, medium, hard)
5. Validiere, ob die Frage wirklich mit dem gegebenen Satz beantwortet werden kann

WICHTIG: Die Frage muss sich spezifisch auf den Inhalt des Satzes beziehen, nicht auf den gesamten Artikel.
"""

        # Definiere das erwartete Ausgabeformat
        expected_response_schema = {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Die generierte Frage"
                },
                "question_type": {
                    "type": "string",
                    "enum": ["rights", "obligations", "purpose", "scope", "definition", "principles", "general"],
                    "description": "Der Typ der Frage"
                },
                "difficulty": {
                    "type": "string",
                    "enum": ["easy", "medium", "hard"],
                    "description": "Die Schwierigkeit der Frage"
                },
                "is_answerable": {
                    "type": "boolean",
                    "description": "Ob die Frage mit dem gegebenen Satz beantwortet werden kann"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Kurze Begründung für die Entscheidung"
                }
            },
            "required": ["question", "question_type", "difficulty", "is_answerable", "reasoning"],
            "additionalProperties": False
        }

        try:
            response = self.client.responses.create(
                model="gpt-4o",
                input=[
                    {
                        "role": "developer",
                        "content": "Du bist ein Experte für die DSGVO und generierst qualitativ hochwertige Fragen für Evaluationsdatensätze."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "qa_pair",
                        "schema": expected_response_schema,
                        "strict": True
                    }
                }
            )

            # Parse die strukturierte Ausgabe
            result = json.loads(response.output_text)

            if result.get('is_answerable', False):
                return {
                    "question": result['question'],
                    "question_type": result['question_type'],
                    "difficulty": result['difficulty'],
                    "reasoning": result['reasoning']
                }
            else:
                logger.debug(f"Frage nicht beantwortbar: {result['reasoning']}")
                return None

        except Exception as e:
            logger.error(f"Fehler bei LLM-Abfrage: {e}")
            return None



    def validate_qa_pair_with_llm(self, question: str, answer: str, context: str) -> bool:
        """Validiert QA-Paar mit GPT-4o"""

        prompt = f"""
Du bist ein Experte für die DSGVO. Validiere folgendes Frage-Antwort-Paar:

Frage: "{question}"
Antwort: "{answer}"
Kontext: "{context}"

Prüfe:
1. Passt die Antwort semantisch zur Frage?
2. Ist die Antwort vollständig genug für die Frage?
3. Kann die Frage mit dem gegebenen Kontext beantwortet werden?

Antworte nur mit "VALID" oder "INVALID" und einer kurzen Begründung.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )

            content = response.choices[0].message.content
            return "VALID" in content.upper()

        except Exception as e:
            logger.error(f"Fehler bei Validierung: {e}")
            return True  # Fallback: Akzeptiere bei Fehlern

    def generate_qa_pairs(self, max_documents: int = None):
        """Generiert QA-Paare mit LLM-Unterstützung (alle Dokumente oder begrenzt auf max_documents)"""
        if max_documents is None:
            logger.info(f"Generiere QA-Paare mit GPT-4o für alle {len(self.documents)} Dokumente...")
            documents_to_process = self.documents
        else:
            logger.info(f"Generiere QA-Paare mit GPT-4o (max. {max_documents} Dokumente)...")
            documents_to_process = self.documents[:max_documents]

        for i, doc in enumerate(documents_to_process):
            logger.info(f"Verarbeite Dokument {i+1}/{len(documents_to_process)}")

            text = doc.get('Text', '').strip()
            if not text or len(text) < 10:
                continue

            article_nr = doc.get('Artikel_nr', 0)
            artikel_name = doc.get('Artikel_Name', '')
            absatz_nr = doc.get('Absatz_nr', 0)
            satz_nr = doc.get('Satz_nr', 0)

            # Hole Artikel-Kontext
            article_context = self.get_article_context(article_nr)

            # Generiere Frage mit LLM
            llm_result = self.generate_question_with_llm(text, article_nr, artikel_name, article_context)

            if not llm_result:
                logger.warning(f"Keine Frage generiert für Dokument {i+1}")
                continue

            # Validiere QA-Paar mit LLM
            if not self.validate_qa_pair_with_llm(llm_result['question'], text, article_context):
                logger.warning(f"QA-Paar validiert nicht für Dokument {i+1}")
                continue

            # Erstelle QA-Paar
            qa_pair = {
                "id": f"dsgvo_art_{article_nr}_abs_{absatz_nr}_satz_{satz_nr}_qa_{len(self.qa_pairs)}",
                "question": llm_result['question'],
                "ground_truth": text,
                "document_id": doc.get('id', ''),
                "question_type": llm_result['question_type'],
                "difficulty": llm_result['difficulty'],
                "metadata": {
                    "kapitel_nr": doc.get('Kapitel_Nr', 0),
                    "kapitel_name": doc.get('Kapitel_Name', ''),
                    "artikel_nr": article_nr,
                    "artikel_name": artikel_name,
                    "absatz_nr": absatz_nr,
                    "unterabsatz_nr": doc.get('Unterabsatz_nr', 0),
                    "satz_nr": satz_nr
                },
                "context": f"DSGVO Artikel {article_nr}: {artikel_name}",
                "article_context": article_context[:500] + "..." if len(article_context) > 500 else article_context,
                "llm_reasoning": llm_result['reasoning']
            }

            self.qa_pairs.append(qa_pair)

                        # Speichere nach jedem erfolgreichen Durchlauf
            logger.info(f"QA-Paar {i+1} erfolgreich generiert und gespeichert")
            self._save_progress()

            # Optional: Pause nach jedem X-ten Dokument um API-Limits zu respektieren
            if (i + 1) % 10 == 0:
                logger.info(f"Pause nach {i+1} Dokumenten...")
                import time
                time.sleep(2)  # 2 Sekunden Pause alle 10 Dokumente

        logger.info(f"Generiert: {len(self.qa_pairs)} QA-Paare")

    def _save_progress(self):
        """Speichert den aktuellen Fortschritt"""
        if not self.qa_pairs:
            return

        # Erstelle temporäre Datei mit aktuellem Stand
        temp_file = Path("data/evaluation/dsgvo_llm_quality_dataset_temp.jsonl")
        temp_file.parent.mkdir(parents=True, exist_ok=True)

        with jsonlines.open(temp_file, 'w') as writer:
            for qa_pair in self.qa_pairs:
                writer.write(qa_pair)

        logger.info(f"Fortschritt gespeichert: {temp_file} ({len(self.qa_pairs)} QA-Paare)")

    def save_dataset(self, output_path: str):
        """Speichert den finalen Datensatz"""
        output_file = Path(output_path)

        # Speichere QA-Paare
        with jsonlines.open(output_file, 'w') as writer:
            for qa_pair in self.qa_pairs:
                writer.write(qa_pair)

        # Erstelle Zusammenfassung
        summary = self._create_summary()
        summary_file = output_file.parent / f"{output_file.stem}_summary.json"

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Lösche temporäre Datei
        temp_file = Path("data/evaluation/dsgvo_llm_quality_dataset_temp.jsonl")
        if temp_file.exists():
            temp_file.unlink()

        logger.info(f"Finaler Datensatz gespeichert: {output_file}")
        logger.info(f"Zusammenfassung gespeichert: {summary_file}")

    def _create_summary(self) -> Dict[str, Any]:
        """Erstellt eine Zusammenfassung des Datensatzes"""

        question_types = defaultdict(int)
        difficulties = defaultdict(int)
        articles_covered = set()

        for qa in self.qa_pairs:
            question_types[qa['question_type']] += 1
            difficulties[qa['difficulty']] += 1
            articles_covered.add(qa['metadata']['artikel_nr'])

        return {
            "total_qa_pairs": len(self.qa_pairs),
            "question_types": dict(question_types),
            "difficulties": dict(difficulties),
            "articles_covered": len(articles_covered),
            "articles_list": sorted(list(articles_covered)),
            "quality_metrics": {
                "avg_answer_length": sum(len(qa['ground_truth']) for qa in self.qa_pairs) / len(self.qa_pairs),
                "context_coverage": len(self.qa_pairs) / len(self.documents) if self.documents else 0
            }
        }

def main():
    """Hauptfunktion"""

    # Konfiguration
    input_file = "data/output/dsgvo_crawled_2025-08-20_1824.jsonl"
    output_file = "data/evaluation/dsgvo_llm_quality_dataset.jsonl"
    max_documents = None  # ← None = alle Dokumente, oder Zahl für Test

    # Prüfe API-Key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY nicht in .env gefunden!")
        return

    # Erstelle Ausgabeverzeichnis
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Generiere Datensatz
    generator = LLMQAGenerator()
    generator.load_documents(input_file)

    if max_documents:
        logger.info(f"TEST-Modus: Verarbeite nur {max_documents} Dokumente")
        generator.generate_qa_pairs(max_documents=max_documents)
    else:
        logger.info("VOLL-Modus: Verarbeite alle Dokumente")
        generator.generate_qa_pairs()  # Alle Dokumente

    generator.save_dataset(output_file)

    # Zeige Zusammenfassung
    summary = generator._create_summary()
    print("\n" + "="*50)
    if max_documents:
        print("LLM-GENERIERTER QUALITÄTS-DATENSATZ (TEST)")
    else:
        print("LLM-GENERIERTER QUALITÄTS-DATENSATZ (VOLL)")
    print("="*50)
    print(f"QA-Paare: {summary['total_qa_pairs']}")
    print(f"Artikel abgedeckt: {summary['articles_covered']}")
    print(f"Fragentypen: {dict(summary['question_types'])}")
    print(f"Schwierigkeitsgrade: {dict(summary['difficulties'])}")
    print(f"Durchschnittliche Antwortlänge: {summary['quality_metrics']['avg_answer_length']:.1f} Zeichen")
    print(f"Kontext-Abdeckung: {summary['quality_metrics']['context_coverage']:.1%}")

if __name__ == "__main__":
    main()
