#!/usr/bin/env python3
"""
Research RAG System - Vereinfachte wissenschaftliche RAG-Evaluation

Diese Klasse bietet:
- Einfache Konfiguration
- Automatische Metriken-Berechnung
- Wissenschaftliche Evaluation
- Komponenten-Vergleich
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Komponenten importieren
from components.chunkers.line_chunker import LineChunker
from components.embeddings.sentence_transformer_embedding import SentenceTransformersEmbedding
from components.vector_stores.in_memory_vector_store import InMemoryVectorStore
from components.language_models.openai_language_model import OpenAILanguageModel
from evaluations.rag_metrics import RAGMetrics
from data_loader import DataLoader


class ResearchRAG:
    """
    Vereinfachtes RAG-System für wissenschaftliche Forschung.

    Features:
    - Ein-Klick-Evaluation
    - Automatische Metriken
    - Komponenten-Vergleich
    - Einfache Konfiguration
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialisiert das Research RAG System.

        Args:
            config: Konfigurationsdictionary mit allen Einstellungen
        """
        self.config = config
        self.data_loader = DataLoader()
        self.metrics = RAGMetrics()

        # Komponenten basierend auf Konfiguration initialisieren
        self._initialize_components()

        # Daten laden
        self._load_data()

        print(f"✅ ResearchRAG initialisiert")
        print(f"   📊 {len(self.documents)} Dokumente geladen")
        print(f"   ❓ {len(self.test_questions)} Test-Fragen bereit")

    def _initialize_components(self):
        """Initialisiert RAG-Komponenten basierend auf Konfiguration."""

        # Chunker
        if self.config.get('chunker') == 'recursive':
            from components.chunkers.recursive_chunker import RecursiveChunker
            self.chunker = RecursiveChunker(
                chunk_size=self.config.get('chunk_size', 1000),
                chunk_overlap=self.config.get('chunk_overlap', 200)
            )
        else:
            self.chunker = LineChunker()

        # Embedding
        if self.config.get('embedding') == 'openai':
            from components.embeddings.openai_embedding import OpenAIEmbedding
            self.embedding = OpenAIEmbedding(api_key=self.config.get('api_key'))
        else:
            self.embedding = SentenceTransformersEmbedding()

        # Vector Store
        if self.config.get('vector_store') == 'chroma':
            from components.vector_stores.chroma_vector_store import ChromaVectorStore
            self.vector_store = ChromaVectorStore()
        else:
            self.vector_store = InMemoryVectorStore()

        # Language Model
        if self.config.get('llm') == 'local':
            from components.language_models.local_language_model import LocalLanguageModel
            self.language_model = LocalLanguageModel()
        else:
            self.language_model = OpenAILanguageModel(api_key=self.config.get('api_key'))

    def _load_data(self):
        """Lädt DSGVO-Daten und Test-Fragen."""
        try:
            self.documents = self.data_loader.load_dsgvo_data()
            self.test_questions = self.data_loader.get_test_questions()

            # Begrenzen für Performance
            num_questions = self.config.get('num_test_questions', 5)
            self.test_questions = self.test_questions[:num_questions]

        except Exception as e:
            print(f"⚠️ Fehler beim Laden der DSGVO-Daten: {e}")
            print("📄 Verwende Fallback-Daten...")

            # Fallback-Daten
            self.documents = [
                "Die DSGVO ist eine EU-Verordnung zum Datenschutz.",
                "Personenbezogene Daten sind alle Informationen über eine Person.",
                "Die Verarbeitung ist nur unter bestimmten Bedingungen erlaubt.",
                "Betroffene haben Rechte wie Auskunft und Löschung."
            ]

            self.test_questions = [
                "Was ist die DSGVO?",
                "Was sind personenbezogene Daten?",
                "Welche Rechte haben Betroffene?",
                "Wann ist Verarbeitung erlaubt?"
            ]

    def run_full_evaluation(self) -> Dict[str, Any]:
        """
        Führt vollständige RAG-Evaluation durch.

        Returns:
            Dictionary mit allen Evaluationsergebnissen
        """
        print("🔄 Starte vollständige RAG-Evaluation...")

        start_time = time.time()

        # 1. Dokumente verarbeiten
        print("   📚 Verarbeite Dokumente...")
        processing_start = time.time()
        self._process_documents()
        processing_time = time.time() - processing_start

        # 2. Queries ausführen und Metriken sammeln
        print("   💬 Führe Test-Queries aus...")
        query_results = self._run_test_queries()

        # 3. Wissenschaftliche Metriken berechnen
        print("   📊 Berechne wissenschaftliche Metriken...")
        scientific_metrics = self._calculate_scientific_metrics(query_results)

        # 4. Performance-Metriken
        total_time = time.time() - start_time
        performance_metrics = {
            'total_evaluation_time': total_time,
            'document_processing_time': processing_time,
            'avg_query_time': query_results['avg_query_time'],
            'queries_per_second': len(self.test_questions) / query_results['total_query_time']
        }

        # 5. Ergebnisse zusammenfassen
        results = {
            'timestamp': datetime.now().isoformat(),
            'configuration': self._get_config_summary(),
            'data_stats': {
                'num_documents': len(self.documents),
                'num_test_questions': len(self.test_questions),
                'total_chunks': query_results.get('total_chunks', 0)
            },
            'scientific_metrics': scientific_metrics,
            'performance_metrics': performance_metrics,
            'sample_qa': query_results['sample_qa'][:3],  # Erste 3 als Beispiel
            'detailed_results': query_results
        }

        print("✅ Evaluation abgeschlossen!")
        return results

    def _process_documents(self):
        """Verarbeitet Dokumente durch die RAG-Pipeline."""
        all_chunks = []
        all_embeddings = []

        for doc in self.documents:
            # Chunking
            chunks = self.chunker.split_document(doc)
            all_chunks.extend(chunks)

            # Embeddings
            embeddings = self.embedding.embed_texts(chunks)
            all_embeddings.extend(embeddings)

        # In Vector Store speichern
        self.vector_store.add_texts(all_chunks, all_embeddings)
        self.total_chunks = len(all_chunks)

    def _run_test_queries(self) -> Dict[str, Any]:
        """Führt alle Test-Queries aus und sammelt Daten."""
        results = {
            'answers': [],
            'query_times': [],
            'retrieved_docs': [],
            'context_relevances': [],
            'answer_relevances': [],
            'sample_qa': []
        }

        for question in self.test_questions:
            # Query ausführen mit Zeitmessung
            start_time = time.time()

            # Ähnliche Dokumente finden
            similar_docs = self.vector_store.similarity_search(
                question,
                k=self.config.get('top_k', 5)
            )
            retrieved_texts = [doc["text"] for doc in similar_docs]

            # Antwort generieren
            try:
                answer = self.language_model.generate_response(question, retrieved_texts)
            except Exception as e:
                answer = f"Fehler bei Antwortgenerierung: {str(e)}"

            query_time = time.time() - start_time

            # RAGAS-Metriken berechnen
            ragas_metrics = self.metrics.calculate_ragas_metrics(
                context=retrieved_texts,
                question=question,
                answer=answer
            )

            # Ergebnisse sammeln
            results['answers'].append(answer)
            results['query_times'].append(query_time)
            results['retrieved_docs'].append(retrieved_texts)
            results['context_relevances'].append(ragas_metrics.get('context_relevance', 0))
            results['answer_relevances'].append(ragas_metrics.get('answer_relevance', 0))
            results['sample_qa'].append({
                'question': question,
                'answer': answer[:200] + "..." if len(answer) > 200 else answer,
                'query_time': query_time
            })

        # Zeitstatistiken
        results['total_query_time'] = sum(results['query_times'])
        results['avg_query_time'] = results['total_query_time'] / len(self.test_questions)
        results['total_chunks'] = self.total_chunks

        return results

    def _calculate_scientific_metrics(self, query_results: Dict[str, Any]) -> Dict[str, Any]:
        """Berechnet wissenschaftliche Evaluationsmetriken."""

        # Simuliere Ground Truth für Retrieval-Evaluation
        # (In echter Forschung würden diese manuell annotiert)
        relevant_docs_per_question = []
        for i, question in enumerate(self.test_questions):
            # Einfache Heuristik: Dokumente mit Schlüsselwörtern sind relevant
            question_words = set(question.lower().split())
            relevant = []

            for doc in self.documents:
                if any(word in doc.lower() for word in question_words if len(word) > 3):
                    relevant.append(doc)

            # Mindestens 3 relevante Dokumente für Evaluation
            if len(relevant) < 3:
                relevant.extend(self.documents[:3])

            relevant_docs_per_question.append(relevant[:5])

        # Retrieval-Metriken berechnen
        retrieval_metrics = []
        for retrieved, relevant in zip(query_results['retrieved_docs'], relevant_docs_per_question):
            metrics = self.metrics.calculate_precision_recall_f1(retrieved, relevant)
            retrieval_metrics.append(metrics)

        # Durchschnittswerte berechnen
        avg_precision = sum(m['precision'] for m in retrieval_metrics) / len(retrieval_metrics)
        avg_recall = sum(m['recall'] for m in retrieval_metrics) / len(retrieval_metrics)
        avg_f1 = sum(m['f1_score'] for m in retrieval_metrics) / len(retrieval_metrics)

        # RAGAS-Durchschnitte
        avg_context_relevance = sum(query_results['context_relevances']) / len(query_results['context_relevances'])
        avg_answer_relevance = sum(query_results['answer_relevances']) / len(query_results['answer_relevances'])

        # Standardabweichungen
        def std_dev(values):
            if len(values) <= 1:
                return 0.0
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            return variance ** 0.5

        precisions = [m['precision'] for m in retrieval_metrics]
        recalls = [m['recall'] for m in retrieval_metrics]
        f1s = [m['f1_score'] for m in retrieval_metrics]

        return {
            'retrieval_metrics': {
                'precision@5': avg_precision,
                'recall@5': avg_recall,
                'f1@5': avg_f1,
                'std_precision@5': std_dev(precisions),
                'std_recall@5': std_dev(recalls),
                'std_f1@5': std_dev(f1s)
            },
            'ragas_metrics': {
                'context_relevance': avg_context_relevance,
                'answer_relevance': avg_answer_relevance,
                'std_context_relevance': std_dev(query_results['context_relevances']),
                'std_answer_relevance': std_dev(query_results['answer_relevances'])
            },
            'detailed_per_question': [
                {
                    'question': q,
                    'precision': m['precision'],
                    'recall': m['recall'],
                    'f1': m['f1_score'],
                    'context_relevance': cr,
                    'answer_relevance': ar
                }
                for q, m, cr, ar in zip(
                    self.test_questions,
                    retrieval_metrics,
                    query_results['context_relevances'],
                    query_results['answer_relevances']
                )
            ]
        }

    def _get_config_summary(self) -> Dict[str, str]:
        """Erstellt Zusammenfassung der Konfiguration."""
        return {
            'chunker': self.config.get('chunker', 'line'),
            'embedding': self.config.get('embedding', 'sentence_transformers'),
            'vector_store': self.config.get('vector_store', 'in_memory'),
            'language_model': self.config.get('llm', 'openai'),
            'chunk_size': str(self.config.get('chunk_size', 'default')),
            'top_k': str(self.config.get('top_k', 5))
        }

    def display_results(self, results: Dict[str, Any]):
        """Zeigt Evaluationsergebnisse übersichtlich an."""
        print("\n" + "="*80)
        print("📊 RAG RESEARCH SYSTEM - EVALUATIONSERGEBNISSE")
        print("="*80)

        # Konfiguration
        print(f"\n⚙️ KONFIGURATION:")
        config = results['configuration']
        print(f"   Chunker: {config['chunker']}")
        print(f"   Embedding: {config['embedding']}")
        print(f"   Vector Store: {config['vector_store']}")
        print(f"   Language Model: {config['language_model']}")
        print(f"   Chunk Size: {config['chunk_size']}")
        print(f"   Top-K: {config['top_k']}")

        # Datenstatistiken
        print(f"\n📄 DATENSTATISTIKEN:")
        data_stats = results['data_stats']
        print(f"   Dokumente: {data_stats['num_documents']}")
        print(f"   Test-Fragen: {data_stats['num_test_questions']}")
        print(f"   Chunks: {data_stats['total_chunks']}")

        # Wissenschaftliche Metriken
        print(f"\n🎯 RETRIEVAL-METRIKEN:")
        retrieval = results['scientific_metrics']['retrieval_metrics']
        print(f"   Precision@5: {retrieval['precision@5']:.3f} (±{retrieval['std_precision@5']:.3f})")
        print(f"   Recall@5:    {retrieval['recall@5']:.3f} (±{retrieval['std_recall@5']:.3f})")
        print(f"   F1@5:        {retrieval['f1@5']:.3f} (±{retrieval['std_f1@5']:.3f})")

        print(f"\n📝 RAGAS-METRIKEN:")
        ragas = results['scientific_metrics']['ragas_metrics']
        print(f"   Context Relevance: {ragas['context_relevance']:.3f} (±{ragas['std_context_relevance']:.3f})")
        print(f"   Answer Relevance:  {ragas['answer_relevance']:.3f} (±{ragas['std_answer_relevance']:.3f})")

        # Performance
        print(f"\n⚡ PERFORMANCE-METRIKEN:")
        perf = results['performance_metrics']
        print(f"   Gesamtzeit: {perf['total_evaluation_time']:.2f}s")
        print(f"   Dokumentverarbeitung: {perf['document_processing_time']:.2f}s")
        print(f"   Ø Zeit/Query: {perf['avg_query_time']:.3f}s")
        print(f"   Queries/Sekunde: {perf['queries_per_second']:.2f}")

        # Beispiel-Antworten
        print(f"\n💬 BEISPIEL-ANTWORTEN:")
        for i, qa in enumerate(results['sample_qa'], 1):
            print(f"   {i}. Q: {qa['question']}")
            print(f"      A: {qa['answer']}")
            print(f"      ⏱️ {qa['query_time']:.3f}s")
            print()

        # Zeitstempel
        print(f"🕐 Evaluation durchgeführt: {results['timestamp']}")
        print("="*80)

    def query_with_metrics(self, question: str) -> tuple[str, Dict[str, Any]]:
        """
        Führt eine Query mit Metriken-Berechnung durch.

        Args:
            question: Die zu beantwortende Frage

        Returns:
            Tuple aus (Antwort, Metriken-Dictionary)
        """
        start_time = time.time()

        # Retrieval
        similar_docs = self.vector_store.similarity_search(question, k=self.config.get('top_k', 5))
        retrieved_texts = [doc["text"] for doc in similar_docs]

        # Antwort generieren
        try:
            answer = self.language_model.generate_response(question, retrieved_texts)
        except Exception as e:
            answer = f"Fehler: {str(e)}"

        response_time = time.time() - start_time

        # Metriken berechnen
        ragas_metrics = self.metrics.calculate_ragas_metrics(
            context=retrieved_texts,
            question=question,
            answer=answer
        )

        metrics = {
            'response_time': response_time,
            'context_relevance': ragas_metrics.get('context_relevance', 0),
            'answer_relevance': ragas_metrics.get('answer_relevance', 0),
            'retrieved_docs': retrieved_texts,
            'num_retrieved': len(retrieved_texts)
        }

        return answer, metrics

    def compare_configurations(self, configurations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Vergleicht verschiedene RAG-Konfigurationen.

        Args:
            configurations: Liste von Konfiguration-Dictionaries

        Returns:
            Vergleichsergebnisse
        """
        print("🧪 Starte Konfigurations-Vergleich...")

        comparison_results = {
            'timestamp': datetime.now().isoformat(),
            'configurations': [],
            'summary': {}
        }

        for i, config_changes in enumerate(configurations):
            print(f"   {i+1}/{len(configurations)}: {config_changes.get('name', f'Config {i+1}')}")

            # Neue Konfiguration erstellen
            new_config = self.config.copy()
            new_config.update(config_changes)

            try:
                # Neues RAG-System für diese Konfiguration
                test_rag = ResearchRAG(new_config)
                results = test_rag.run_full_evaluation()

                comparison_results['configurations'].append({
                    'name': config_changes.get('name', f'Config {i+1}'),
                    'config': config_changes,
                    'results': results,
                    'status': 'success'
                })

            except Exception as e:
                comparison_results['configurations'].append({
                    'name': config_changes.get('name', f'Config {i+1}'),
                    'config': config_changes,
                    'error': str(e),
                    'status': 'failed'
                })

        print("✅ Vergleich abgeschlossen!")
        return comparison_results

    def display_comparison(self, comparison: Dict[str, Any]):
        """Zeigt Vergleichsergebnisse übersichtlich an."""
        print("\n" + "="*80)
        print("🔬 KONFIGURATIONS-VERGLEICH")
        print("="*80)

        successful_configs = [c for c in comparison['configurations'] if c['status'] == 'success']

        if not successful_configs:
            print("❌ Keine erfolgreichen Konfigurationen!")
            return

        # Vergleichstabelle
        print(f"\n📊 METRIKEN-VERGLEICH:")
        print(f"{'Konfiguration':<20} {'Precision@5':<12} {'Recall@5':<10} {'F1@5':<8} {'Context Rel':<12} {'Q/s':<8}")
        print("-" * 80)

        for config in successful_configs:
            name = config['name'][:18]
            results = config['results']
            retrieval = results['scientific_metrics']['retrieval_metrics']
            ragas = results['scientific_metrics']['ragas_metrics']
            perf = results['performance_metrics']

            print(f"{name:<20} {retrieval['precision@5']:<12.3f} {retrieval['recall@5']:<10.3f} "
                  f"{retrieval['f1@5']:<8.3f} {ragas['context_relevance']:<12.3f} {perf['queries_per_second']:<8.2f}")

        # Beste Konfiguration
        best_f1 = max(successful_configs,
                     key=lambda x: x['results']['scientific_metrics']['retrieval_metrics']['f1@5'])

        print(f"\n🏆 BESTE KONFIGURATION (F1@5): {best_f1['name']}")
        print(f"   F1-Score: {best_f1['results']['scientific_metrics']['retrieval_metrics']['f1@5']:.3f}")

        print("="*80)

    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Speichert Evaluationsergebnisse als JSON.

        Args:
            results: Ergebnisse-Dictionary
            filename: Optionaler Dateiname

        Returns:
            Pfad zur gespeicherten Datei
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_evaluation_{timestamp}.json"

        # Results-Verzeichnis erstellen
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        filepath = results_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return str(filepath)