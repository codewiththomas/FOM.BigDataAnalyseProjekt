import os
from typing import List, Dict, Any, Optional
import re


class DataLoader:
    """
    Data Loader für verschiedene Dokumenttypen, insbesondere DSGVO-Daten.
    """

    def __init__(self, data_directory: str = "data"):
        self.data_directory = data_directory

    def load_dsgvo_document(self, filepath: Optional[str] = None) -> str:
        """
        Lädt das DSGVO-Dokument und bereinigt es.
        """
        if filepath is None:
            filepath = os.path.join(self.data_directory, "raw", "dsgvo.txt")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Bereinige den Text
            cleaned_content = self._clean_dsgvo_text(content)
            return cleaned_content

        except FileNotFoundError:
            raise FileNotFoundError(f"DSGVO-Datei nicht gefunden: {filepath}")
        except Exception as e:
            raise Exception(f"Fehler beim Laden der DSGVO-Datei: {str(e)}")

    def _clean_dsgvo_text(self, text: str) -> str:
        """
        Bereinigt den DSGVO-Text von unnötigen Formatierungen.
        """
        # Entferne übermäßige Leerzeichen und Zeilenumbrüche
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)

        # Entferne Seitenzahlen und andere Formatierungen
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)

        # Bereinige Artikel-Nummern
        text = re.sub(r'Artikel\s+(\d+)\s*', r'Artikel \1: ', text)

        # Entferne leere Zeilen am Anfang und Ende
        text = text.strip()

        return text

    def split_dsgvo_into_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Teilt das DSGVO-Dokument in logische Abschnitte auf.
        """
        sections = []

        # Teile nach Artikeln auf
        article_pattern = r'Artikel\s+(\d+)[:\s]*([^\n]+)'
        articles = re.finditer(article_pattern, text, re.IGNORECASE)

        current_section = ""
        current_article = ""

        for match in articles:
            article_num = match.group(1)
            article_title = match.group(2).strip()

            # Speichere vorherigen Abschnitt
            if current_section:
                sections.append({
                    "type": "article",
                    "number": current_article,
                    "title": "",
                    "content": current_section.strip()
                })

            current_article = article_num
            current_section = match.group(0) + "\n"

        # Letzten Abschnitt hinzufügen
        if current_section:
            sections.append({
                "type": "article",
                "number": current_article,
                "title": "",
                "content": current_section.strip()
            })

        return sections

    def create_dsgvo_qa_dataset(self, text: str) -> List[Dict[str, str]]:
        """
        Erstellt ein Q&A-Dataset basierend auf dem DSGVO-Text.
        """
        qa_pairs = []

        # Beispiel-Fragen für DSGVO
        questions = [
            "Was ist die DSGVO?",
            "Welche Rechte haben betroffene Personen?",
            "Was ist ein Verantwortlicher?",
            "Was ist ein Auftragsverarbeiter?",
            "Was ist die Einwilligung?",
            "Was sind personenbezogene Daten?",
            "Was ist das Recht auf Löschung?",
            "Was ist das Recht auf Datenübertragbarkeit?",
            "Was ist die Datenschutz-Folgenabschätzung?",
            "Was sind die Grundsätze der Datenverarbeitung?"
        ]

        # Einfache Antworten basierend auf Schlüsselwörtern
        for question in questions:
            answer = self._generate_simple_answer(question, text)
            qa_pairs.append({
                "question": question,
                "answer": answer,
                "context": self._find_relevant_context(question, text)
            })

        return qa_pairs

    def _generate_simple_answer(self, question: str, text: str) -> str:
        """
        Generiert eine einfache Antwort basierend auf Schlüsselwörtern.
        """
        question_lower = question.lower()

        if "dsgvo" in question_lower or "datenschutz-grundverordnung" in question_lower:
            return "Die DSGVO (Datenschutz-Grundverordnung) ist eine EU-Verordnung zum Schutz personenbezogener Daten."

        elif "rechte" in question_lower and "betroffene" in question_lower:
            return "Betroffene Personen haben das Recht auf Auskunft, Berichtigung, Löschung und Datenübertragbarkeit."

        elif "verantwortlicher" in question_lower:
            return "Ein Verantwortlicher ist die natürliche oder juristische Person, die über die Zwecke und Mittel der Verarbeitung entscheidet."

        elif "auftragsverarbeiter" in question_lower:
            return "Ein Auftragsverarbeiter ist eine natürliche oder juristische Person, die personenbezogene Daten im Auftrag des Verantwortlichen verarbeitet."

        elif "einwilligung" in question_lower:
            return "Die Einwilligung ist eine freiwillige, informierte und unmissverständliche Willensbekundung der betroffenen Person."

        elif "personenbezogene daten" in question_lower:
            return "Personenbezogene Daten sind alle Informationen, die sich auf eine identifizierte oder identifizierbare natürliche Person beziehen."

        else:
            return "Diese Frage kann basierend auf dem DSGVO-Text beantwortet werden."

    def _find_relevant_context(self, question: str, text: str) -> str:
        """
        Findet relevanten Kontext für eine Frage.
        """
        # Einfache Schlüsselwort-basierte Suche
        question_lower = question.lower()

        if "dsgvo" in question_lower:
            return text[:1000]  # Erste 1000 Zeichen

        elif "rechte" in question_lower:
            # Suche nach Artikeln über Rechte
            rights_pattern = r'Artikel\s+\d+[^.]*Recht[^.]*\.'
            matches = re.findall(rights_pattern, text, re.IGNORECASE)
            return matches[0] if matches else text[:500]

        else:
            return text[:500]  # Erste 500 Zeichen als Fallback

    def load_test_questions(self) -> List[str]:
        """
        Lädt eine Liste von Testfragen für die DSGVO.
        """
        return [
            "Was ist die DSGVO?",
            "Welche Rechte haben betroffene Personen?",
            "Was ist ein Verantwortlicher?",
            "Was ist ein Auftragsverarbeiter?",
            "Was ist die Einwilligung?",
            "Was sind personenbezogene Daten?",
            "Was ist das Recht auf Löschung?",
            "Was ist das Recht auf Datenübertragbarkeit?",
            "Was ist die Datenschutz-Folgenabschätzung?",
            "Was sind die Grundsätze der Datenverarbeitung?",
            "Was ist die Pseudonymisierung?",
            "Was ist die Anonymisierung?",
            "Was ist die Datenminimierung?",
            "Was ist die Zweckbindung?",
            "Was ist die Speicherbegrenzung?",
            "Was ist die Richtigkeit?",
            "Was ist die Integrität und Vertraulichkeit?",
            "Was ist die Rechenschaftspflicht?",
            "Was ist die Aufsichtsbehörde?",
            "Was ist die Datenschutz-Folgenabschätzung?"
        ]

    def get_dsgvo_statistics(self, text: str) -> Dict[str, Any]:
        """
        Berechnet Statistiken über das DSGVO-Dokument.
        """
        return {
            "total_characters": len(text),
            "total_words": len(text.split()),
            "total_lines": len(text.split('\n')),
            "article_count": len(re.findall(r'Artikel\s+\d+', text)),
            "paragraph_count": len(re.findall(r'\(\d+\)', text))
        }

    def load_dsgvo_data(self) -> List[str]:
        """
        Einfache API für Notebook: Lädt DSGVO-Daten und gibt Liste von Dokumenten zurück.
        """
        try:
            # Lade das komplette DSGVO-Dokument
            full_text = self.load_dsgvo_document()

            # Teile in Abschnitte auf
            sections = self.split_dsgvo_into_sections(full_text)

            # Konvertiere zu einfacher Liste von Strings
            documents = []
            for section in sections:
                if section.get('content'):
                    documents.append(section['content'])

            # Falls keine Abschnitte gefunden, teile einfach in Chunks
            if not documents:
                # Teile in 1000-Zeichen-Chunks
                chunk_size = 1000
                for i in range(0, len(full_text), chunk_size):
                    chunk = full_text[i:i + chunk_size]
                    if chunk.strip():
                        documents.append(chunk.strip())

            return documents

        except Exception as e:
            print(f"Fehler beim Laden der DSGVO-Daten: {e}")
            return []

    def get_test_questions(self) -> List[str]:
        """
        Einfache API für Notebook: Gibt Liste von Testfragen zurück.
        """
        return self.load_test_questions()

    def load_legal_document(self, document_type: str, filepath: Optional[str] = None) -> List[str]:
        """
        Erweiterte API für andere Rechtstexte in der Zukunft.
        """
        if document_type.lower() == "dsgvo":
            return self.load_dsgvo_data()
        elif document_type.lower() == "ggb":
            # Placeholder für Grundgesetz
            if filepath:
                return self._load_generic_legal_document(filepath)
        elif document_type.lower() == "stgb":
            # Placeholder für Strafgesetzbuch
            if filepath:
                return self._load_generic_legal_document(filepath)

        return []

    def _load_generic_legal_document(self, filepath: str) -> List[str]:
        """
        Generische Methode für andere Rechtstexte.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Einfache Paragraph-basierte Aufteilung
            paragraphs = content.split('\n\n')
            documents = []

            for para in paragraphs:
                para = para.strip()
                if para and len(para) > 50:  # Nur substantielle Paragraphen
                    documents.append(para)

            return documents

        except Exception as e:
            print(f"Fehler beim Laden des Dokuments {filepath}: {e}")
            return []