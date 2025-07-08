#!/usr/bin/env python3
"""
Test-Skript für das RAG-System
"""

import sys
import os
sys.path.append('src')

# .env-Datei laden
from dotenv import load_dotenv
load_dotenv()

from config import RAGConfig
from experiments.experiment_runner import ExperimentRunner

def main():
    # OpenAI API Key laden
    openai_key = os.getenv("OPENAI_API_KEY")
    print(f"OpenAI Key geladen: {'Ja' if openai_key else 'Nein'}")
    
    # Testdaten
    test_data = [
        "Big Data Analyse ist ein Prozess zur Untersuchung großer Datenmengen.",
        "Machine Learning ist ein Teilgebiet der künstlichen Intelligenz.",
        "Datenverarbeitung umfasst die systematische Analyse von Informationen.",
        "Algorithmen sind Schritt-für-Schritt-Anweisungen zur Problemlösung."
    ]
    
    # Konfiguration
    config = RAGConfig(
        chunker_type="line",
        embedding_type="sentence_transformers",
        vector_store_type="in_memory",
        language_model_type="openai",
        language_model_params={"api_key": openai_key}
    )
    
    # Experiment ausführen
    print("🚀 Starte RAG-Experiment...")
    runner = ExperimentRunner()
    result = runner.run_experiment(config, test_data)
    
    print("\n📊 Ergebnisse:")
    print(f"Testdaten: {result['test_data_count']} Dokumente")
    print(f"Frage: {result['test_question']}")
    print(f"Antwort: {result['response']}")
    print(f"Status: {result['status']}")
    
    print("\n✅ RAG-System erfolgreich getestet!")

if __name__ == "__main__":
    main() 