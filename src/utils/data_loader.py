import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path


class DataLoader:
    """
    Utility-Klasse für das Laden und Verarbeiten von Dokumenten.
    
    Unterstützt verschiedene Dateiformate und bietet Funktionen für
    das Laden von DSGVO-Dokumenten und QA-Datensätzen.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialisiert den Data Loader.
        
        Args:
            data_dir: Basis-Verzeichnis für Daten
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.evaluation_dir = self.data_dir / "evaluation"
    
    def load_dsgvo_document(self, filename: str = "dsgvo.txt") -> str:
        """
        Lädt das DSGVO-Dokument.
        
        Args:
            filename: Name der DSGVO-Datei
            
        Returns:
            DSGVO-Text als String
        """
        file_path = self.raw_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"DSGVO-Datei nicht gefunden: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return content.strip()
            
        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden der DSGVO-Datei: {e}")
    
    def load_qa_dataset(self, filename: str = "qa_pairs.json") -> List[Dict[str, Any]]:
        """
        Lädt einen QA-Datensatz.
        
        Args:
            filename: Name der QA-Datei
            
        Returns:
            Liste von QA-Paaren
        """
        file_path = self.evaluation_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"QA-Datei nicht gefunden: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validierung der QA-Struktur
            if "questions" not in data:
                raise ValueError("QA-Datei muss 'questions' Feld enthalten")
            
            return data["questions"]
            
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Fehler beim Parsen der QA-Datei: {e}")
        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden der QA-Datei: {e}")
    
    def save_qa_dataset(self, qa_pairs: List[Dict[str, Any]], 
                       filename: str = "qa_pairs.json", 
                       metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Speichert einen QA-Datensatz.
        
        Args:
            qa_pairs: Liste von QA-Paaren
            filename: Name der Zieldatei
            metadata: Optionale Metadaten
        """
        file_path = self.evaluation_dir / filename
        
        # Verzeichnis erstellen falls nicht vorhanden
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "metadata": metadata or {
                "version": "1.0.0",
                "description": "DSGVO QA-Datensatz",
                "total_questions": len(qa_pairs)
            },
            "questions": qa_pairs
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            raise RuntimeError(f"Fehler beim Speichern der QA-Datei: {e}")
    
    def create_sample_qa_dataset(self) -> List[Dict[str, Any]]:
        """
        Erstellt einen Beispiel-QA-Datensatz für die DSGVO.
        
        Returns:
            Liste von Beispiel-QA-Paaren
        """
        sample_qa_pairs = [
            {
                "id": "dsgvo_001",
                "question": "Was ist die maximale Geldbuße nach Art. 83 DSGVO?",
                "category": "faktenfragen",
                "difficulty": "easy",
                "gold_answer": "20 Millionen Euro oder 4% des weltweiten Jahresumsatzes, je nachdem welcher Betrag höher ist.",
                "relevant_articles": ["Art. 83"],
                "keywords": ["Geldbuße", "Sanktionen", "Artikel 83"]
            },
            {
                "id": "dsgvo_002", 
                "question": "Welche Rechtsgrundlagen für die Verarbeitung personenbezogener Daten gibt es?",
                "category": "faktenfragen",
                "difficulty": "medium",
                "gold_answer": "Die Rechtsgrundlagen sind in Art. 6 DSGVO aufgeführt: Einwilligung, Vertragserfüllung, rechtliche Verpflichtung, Schutz lebenswichtiger Interessen, öffentliche Aufgabe und berechtigte Interessen.",
                "relevant_articles": ["Art. 6"],
                "keywords": ["Rechtsgrundlagen", "Verarbeitung", "Artikel 6"]
            },
            {
                "id": "dsgvo_003",
                "question": "Wie läuft das Verfahren bei einer Datenschutz-Folgenabschätzung ab?",
                "category": "prozessfragen",
                "difficulty": "hard",
                "gold_answer": "Eine Datenschutz-Folgenabschätzung nach Art. 35 DSGVO umfasst die Beschreibung der Verarbeitung, Bewertung der Notwendigkeit und Verhältnismäßigkeit, Risikobewertung und Schutzmaßnahmen.",
                "relevant_articles": ["Art. 35"],
                "keywords": ["Datenschutz-Folgenabschätzung", "DSFA", "Artikel 35"]
            },
            {
                "id": "dsgvo_004",
                "question": "Welche Rechte haben betroffene Personen?",
                "category": "faktenfragen", 
                "difficulty": "medium",
                "gold_answer": "Betroffene Personen haben verschiedene Rechte: Auskunft (Art. 15), Berichtigung (Art. 16), Löschung (Art. 17), Einschränkung (Art. 18), Datenübertragbarkeit (Art. 20) und Widerspruch (Art. 21).",
                "relevant_articles": ["Art. 15", "Art. 16", "Art. 17", "Art. 18", "Art. 20", "Art. 21"],
                "keywords": ["Betroffenenrechte", "Auskunft", "Löschung", "Berichtigung"]
            },
            {
                "id": "dsgvo_005",
                "question": "Was sind die Grundsätze der Datenverarbeitung?",
                "category": "faktenfragen",
                "difficulty": "medium", 
                "gold_answer": "Die Grundsätze nach Art. 5 DSGVO sind: Rechtmäßigkeit, Verarbeitung nach Treu und Glauben, Transparenz, Zweckbindung, Datenminimierung, Richtigkeit, Speicherbegrenzung, Integrität und Vertraulichkeit sowie Rechenschaftspflicht.",
                "relevant_articles": ["Art. 5"],
                "keywords": ["Grundsätze", "Artikel 5", "Datenverarbeitung"]
            }
        ]
        
        return sample_qa_pairs
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Gibt Informationen über eine Datei zurück.
        
        Args:
            file_path: Pfad zur Datei
            
        Returns:
            Dictionary mit Datei-Informationen
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
        
        stat = path.stat()
        
        return {
            "name": path.name,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": stat.st_mtime,
            "extension": path.suffix,
            "absolute_path": str(path.absolute())
        }
    
    def list_available_files(self) -> Dict[str, List[str]]:
        """
        Listet alle verfügbaren Dateien in den Datenverzeichnissen auf.
        
        Returns:
            Dictionary mit Dateien nach Verzeichnis
        """
        files = {
            "raw": [],
            "processed": [],
            "evaluation": []
        }
        
        for dir_name, dir_path in [
            ("raw", self.raw_dir),
            ("processed", self.processed_dir), 
            ("evaluation", self.evaluation_dir)
        ]:
            if dir_path.exists():
                files[dir_name] = [f.name for f in dir_path.iterdir() if f.is_file()]
        
        return files 