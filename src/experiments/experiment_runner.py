from typing import List, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RAGConfig
from rag_system import RAGSystem

class ExperimentRunner:
    def __init__(self):
        self.results = []

    def run_experiment(self, config: RAGConfig, test_data: List[str]) -> dict:
        """Führt ein Experiment mit gegebener Konfiguration aus"""
        print(f"Experiment mit Konfiguration: {config}")
        print(f"Testdaten: {len(test_data)} Dokumente")

        # RAG-System erstellen und testen
        rag_system = RAGSystem(config)
        
        # Dokumente verarbeiten
        rag_system.process_documents(test_data)
        
        # Test-Query ausführen
        test_question = "Was ist Big Data Analyse?"
        response = rag_system.query(test_question)
        
        return {
            "config": config,
            "test_data_count": len(test_data),
            "test_question": test_question,
            "response": response,
            "status": "completed"
        }

    def compare_configurations(self, configs: List[RAGConfig], test_data: List[str]):
        """Vergleicht mehrere Konfigurationen"""
        results = []
        for config in configs:
            result = self.run_experiment(config, test_data)
            results.append(result)
        return results