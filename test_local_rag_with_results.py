#!/usr/bin/env python3
"""
Test-Skript für lokales ResearchRAG-System mit Ergebnis-Speicherung
Testet die lokalen Komponenten und speichert alle Ergebnisse in data/evaluation/results/
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Pfad zum src-Verzeichnis hinzufügen
sys.path.insert(0, 'src')

def test_local_rag_with_results():
    print("🚀 Teste lokales ResearchRAG-System mit Ergebnis-Speicherung...")

    # Ergebnis-Verzeichnis erstellen
    results_dir = Path("data/evaluation/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Zeitstempel für eindeutige Dateinamen
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ergebnis-Struktur initialisieren
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "test_type": "local_rag_retrieval",
            "system_config": {},
            "performance_metrics": {},
            "test_questions": []
        },
        "results": {
            "system_status": "unknown",
            "indexing_results": {},
            "retrieval_results": [],
            "overall_performance": {}
        }
    }

    try:
        # Imports
        from config.pipeline_configs import get_local_config
        from core.rag_pipeline import RAGPipeline

        print("✅ Module erfolgreich importiert")

        # Lokale Konfiguration laden
        config = get_local_config()

        # OpenAI als Fallback für LLM verwenden
        config_dict = config._config.copy()
        config_dict["language_model"] = {
            "type": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 500
        }

        from config.pipeline_configs import PipelineConfig
        config = PipelineConfig(config_dict)

        # Konfiguration in Ergebnisse speichern
        results["metadata"]["system_config"] = config.get_component_types()

        print(f"✅ Konfiguration geladen: {config.get_component_types()}")

        # Pipeline erstellen
        print("🔧 Erstelle Pipeline...")
        start_time = time.time()
        pipeline = RAGPipeline(config)
        pipeline_creation_time = time.time() - start_time

        results["metadata"]["performance_metrics"]["pipeline_creation_time"] = pipeline_creation_time

        print("✅ Pipeline erstellt")

        # Test-Dokument laden
        print("📄 Lade DSGVO-Dokument...")
        dsgvo_path = Path("data/raw/dsgvo.txt")
        if not dsgvo_path.exists():
            print(f"❌ DSGVO-Datei nicht gefunden: {dsgvo_path}")
            results["results"]["system_status"] = "error"
            results["results"]["error"] = f"DSGVO file not found: {dsgvo_path}"
            return False

        with open(dsgvo_path, 'r', encoding='utf-8') as f:
            dsgvo_text = f.read()

        document_length = len(dsgvo_text)
        results["metadata"]["performance_metrics"]["document_length"] = document_length

        print(f"✅ Dokument geladen: {document_length} Zeichen")

        # Dokument indexieren
        print("🔍 Indexiere Dokument...")
        start_time = time.time()
        pipeline.index_documents([dsgvo_text])
        indexing_time = time.time() - start_time

        # Indexierungs-Ergebnisse speichern
        chunk_count = len(pipeline.vector_store._texts) if hasattr(pipeline.vector_store, '_texts') else 0
        avg_chunk_length = document_length / chunk_count if chunk_count > 0 else 0
        chunks_per_second = chunk_count / indexing_time if indexing_time > 0 else 0

        results["results"]["indexing_results"] = {
            "indexing_time": indexing_time,
            "chunk_count": chunk_count,
            "avg_chunk_length": avg_chunk_length,
            "chunks_per_second": chunks_per_second,
            "embedding_dimension": pipeline.embedding.get_embedding_dimension() if hasattr(pipeline.embedding, 'get_embedding_dimension') else None
        }

        print("✅ Dokument indexiert")

        # Test-Fragen laden
        qa_pairs_path = Path("data/evaluation/qa_pairs.json")
        if qa_pairs_path.exists():
            with open(qa_pairs_path, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            test_questions = qa_data.get("questions", [])[:5]  # Erste 5 Fragen
        else:
            # Fallback Test-Fragen
            test_questions = [
                {
                    "id": "test_001",
                    "question": "Was ist die maximale Geldbuße nach der DSGVO?",
                    "category": "faktenfragen",
                    "difficulty": "easy"
                },
                {
                    "id": "test_002",
                    "question": "Welche Rechte haben betroffene Personen?",
                    "category": "faktenfragen",
                    "difficulty": "medium"
                },
                {
                    "id": "test_003",
                    "question": "Was ist eine Datenschutz-Folgenabschätzung?",
                    "category": "prozessfragen",
                    "difficulty": "hard"
                }
            ]

        results["metadata"]["test_questions"] = test_questions

        print(f"\n🧪 Teste Retrieval mit {len(test_questions)} Fragen...")

        # Retrieval-Tests durchführen
        for i, question_data in enumerate(test_questions, 1):
            question = question_data["question"]
            question_id = question_data.get("id", f"q_{i}")

            print(f"\n--- Frage {i}: {question} ---")

            start_time = time.time()

            # Embedding für Frage erstellen
            question_embedding = pipeline.embedding.embed_texts([question])

            # Ähnliche Chunks finden
            similar_chunks = pipeline.vector_store.similarity_search(
                question_embedding[0], top_k=5
            )

            retrieval_time = time.time() - start_time

            print(f"✅ {len(similar_chunks)} relevante Chunks gefunden")

            # Ergebnis für diese Frage speichern
            question_result = {
                "question_id": question_id,
                "question": question,
                "category": question_data.get("category", "unknown"),
                "difficulty": question_data.get("difficulty", "unknown"),
                "retrieval_time": retrieval_time,
                "chunks_found": len(similar_chunks),
                "top_chunks": []
            }

            # Top 3 Chunks mit Details speichern
            for j, chunk in enumerate(similar_chunks[:3]):
                chunk_info = {
                    "rank": j + 1,
                    "similarity_score": float(chunk.get('score', 0)),
                    "chunk_id": chunk.get('id', f'chunk_{j}'),
                    "chunk_text": chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text'],
                    "chunk_length": len(chunk['text']),
                    "metadata": chunk.get('metadata', {})
                }
                question_result["top_chunks"].append(chunk_info)

                # Chunk-Vorschau ausgeben
                chunk_preview = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                print(f"  📝 Chunk {j+1}: {chunk_preview}")
                print(f"     Ähnlichkeit: {chunk.get('score', 'N/A'):.3f}")

            results["results"]["retrieval_results"].append(question_result)

        # Gesamt-Performance berechnen
        total_retrieval_time = sum(r["retrieval_time"] for r in results["results"]["retrieval_results"])
        avg_retrieval_time = total_retrieval_time / len(results["results"]["retrieval_results"])
        avg_similarity_score = sum(
            chunk["similarity_score"] for result in results["results"]["retrieval_results"]
            for chunk in result["top_chunks"][:1]  # Nur beste Treffer
        ) / len(results["results"]["retrieval_results"])

        results["results"]["overall_performance"] = {
            "total_retrieval_time": total_retrieval_time,
            "avg_retrieval_time": avg_retrieval_time,
            "avg_similarity_score": avg_similarity_score,
            "questions_tested": len(test_questions)
        }

        results["results"]["system_status"] = "success"

        print("\n✅ Lokales RAG-System funktioniert!")
        print("\n📊 System-Status:")
        print(f"  - Chunker: {config.get_component_types()['chunker']}")
        print(f"  - Embedding: {config.get_component_types()['embedding']}")
        print(f"  - Vector Store: {config.get_component_types()['vector_store']}")
        print(f"  - Chunks im Index: {chunk_count}")
        print(f"  - Durchschnittliche Retrieval-Zeit: {avg_retrieval_time:.3f}s")
        print(f"  - Durchschnittliche Ähnlichkeit: {avg_similarity_score:.3f}")

        # Ergebnisse speichern
        result_file = results_dir / f"local_rag_test_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Ergebnisse gespeichert in: {result_file}")

        # Zusätzlich: Kurzbericht erstellen
        report_file = results_dir / f"report_{timestamp}.md"
        create_report(results, report_file)
        print(f"📄 Bericht erstellt: {report_file}")

        return True

    except Exception as e:
        print(f"❌ Fehler: {e}")
        results["results"]["system_status"] = "error"
        results["results"]["error"] = str(e)

        # Auch Fehler-Ergebnisse speichern
        result_file = results_dir / f"local_rag_error_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        import traceback
        traceback.print_exc()
        return False

def create_report(results, report_file):
    """Erstellt einen Markdown-Bericht der Testergebnisse."""

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# ResearchRAG Lokaler Test - Bericht\n\n")
        f.write(f"**Zeitstempel:** {results['metadata']['timestamp']}\n\n")

        # System-Konfiguration
        f.write("## System-Konfiguration\n\n")
        config = results['metadata']['system_config']
        for component, type_name in config.items():
            f.write(f"- **{component.title()}:** {type_name}\n")
        f.write("\n")

        # Performance-Metriken
        f.write("## Performance-Metriken\n\n")
        perf = results['metadata']['performance_metrics']
        idx_results = results['results']['indexing_results']

        f.write(f"- **Dokument-Länge:** {perf.get('document_length', 'N/A'):,} Zeichen\n")
        f.write(f"- **Pipeline-Erstellung:** {perf.get('pipeline_creation_time', 0):.2f}s\n")
        f.write(f"- **Indexierungs-Zeit:** {idx_results.get('indexing_time', 0):.2f}s\n")
        f.write(f"- **Chunks erstellt:** {idx_results.get('chunk_count', 0)}\n")
        f.write(f"- **Durchschnittliche Chunk-Länge:** {idx_results.get('avg_chunk_length', 0):.0f} Zeichen\n")
        f.write(f"- **Chunks pro Sekunde:** {idx_results.get('chunks_per_second', 0):.1f}\n")
        f.write(f"- **Embedding-Dimension:** {idx_results.get('embedding_dimension', 'N/A')}\n\n")

        # Retrieval-Ergebnisse
        f.write("## Retrieval-Ergebnisse\n\n")
        overall = results['results']['overall_performance']
        f.write(f"- **Fragen getestet:** {overall.get('questions_tested', 0)}\n")
        f.write(f"- **Gesamt-Retrieval-Zeit:** {overall.get('total_retrieval_time', 0):.2f}s\n")
        f.write(f"- **Durchschnittliche Retrieval-Zeit:** {overall.get('avg_retrieval_time', 0):.3f}s\n")
        f.write(f"- **Durchschnittliche Ähnlichkeit:** {overall.get('avg_similarity_score', 0):.3f}\n\n")

        # Detaillierte Fragen-Ergebnisse
        f.write("## Detaillierte Ergebnisse\n\n")
        for result in results['results']['retrieval_results']:
            f.write(f"### {result['question_id']}: {result['question']}\n\n")
            f.write(f"- **Kategorie:** {result['category']}\n")
            f.write(f"- **Schwierigkeit:** {result['difficulty']}\n")
            f.write(f"- **Retrieval-Zeit:** {result['retrieval_time']:.3f}s\n")
            f.write(f"- **Chunks gefunden:** {result['chunks_found']}\n\n")

            f.write("**Top 3 Chunks:**\n\n")
            for chunk in result['top_chunks'][:3]:
                f.write(f"{chunk['rank']}. **Ähnlichkeit:** {chunk['similarity_score']:.3f}\n")
                f.write(f"   ```\n   {chunk['chunk_text'][:200]}...\n   ```\n\n")

        # Status
        f.write("## System-Status\n\n")
        status = results['results']['system_status']
        if status == "success":
            f.write("✅ **Test erfolgreich abgeschlossen**\n\n")
        else:
            f.write("❌ **Test fehlgeschlagen**\n\n")
            if 'error' in results['results']:
                f.write(f"**Fehler:** {results['results']['error']}\n\n")

if __name__ == "__main__":
    success = test_local_rag_with_results()
    if success:
        print("\n🎉 Test erfolgreich abgeschlossen!")
        print("\nErgebnisse wurden in data/evaluation/results/ gespeichert.")
    else:
        print("\n❌ Test fehlgeschlagen!")
        sys.exit(1)