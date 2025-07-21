#!/usr/bin/env python3
"""
Test-Skript für lokales ResearchRAG-System ohne Ollama
Testet nur die lokalen Komponenten (Chunking, Embedding, Vector Store)
"""

import sys
import os
import json
from pathlib import Path

# Pfad zum src-Verzeichnis hinzufügen
sys.path.insert(0, 'src')

def test_local_rag_no_ollama():
    print("🚀 Teste lokales ResearchRAG-System (ohne Ollama)...")

    try:
        # Imports
        from config.pipeline_configs import get_local_config
        from core.rag_pipeline import RAGPipeline

        print("✅ Module erfolgreich importiert")

        # Lokale Konfiguration laden und für OpenAI-Fallback anpassen
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

        print(f"✅ Konfiguration geladen: {config.get_component_types()}")

        # Pipeline erstellen
        print("🔧 Erstelle Pipeline...")
        pipeline = RAGPipeline(config)
        print("✅ Pipeline erstellt")

        # Test-Dokument laden
        print("📄 Lade DSGVO-Dokument...")
        dsgvo_path = Path("data/raw/dsgvo.txt")
        if not dsgvo_path.exists():
            print(f"❌ DSGVO-Datei nicht gefunden: {dsgvo_path}")
            return False

        with open(dsgvo_path, 'r', encoding='utf-8') as f:
            dsgvo_text = f.read()

        print(f"✅ Dokument geladen: {len(dsgvo_text)} Zeichen")

        # Dokument indexieren
        print("🔍 Indexiere Dokument...")
        pipeline.index_documents([dsgvo_text])
        print("✅ Dokument indexiert")

        # Test-Fragen
        test_questions = [
            "Was ist die maximale Geldbuße nach der DSGVO?",
            "Welche Rechte haben betroffene Personen?",
            "Was ist eine Datenschutz-Folgenabschätzung?"
        ]

        print("\n🧪 Teste Retrieval (ohne Generation)...")

        # Nur Retrieval testen (ohne LLM)
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- Frage {i}: {question} ---")

            # Embedding für Frage erstellen
            question_embedding = pipeline.embedding.embed_texts([question])

            # Ähnliche Chunks finden
            similar_chunks = pipeline.vector_store.similarity_search(
                question_embedding[0], top_k=3
            )

            print(f"✅ {len(similar_chunks)} relevante Chunks gefunden")

            # Erste 2 Chunks anzeigen
            for j, chunk in enumerate(similar_chunks[:2]):
                chunk_text = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                print(f"  📝 Chunk {j+1}: {chunk_text}")
                print(f"     Ähnlichkeit: {chunk.get('score', 'N/A'):.3f}")

        print("\n✅ Lokales RAG-System funktioniert!")
        print("\n📊 System-Status:")
        print(f"  - Chunker: {config.get_component_types()['chunker']}")
        print(f"  - Embedding: {config.get_component_types()['embedding']}")
        print(f"  - Vector Store: {config.get_component_types()['vector_store']}")
        print(f"  - Chunks im Index: {len(pipeline.vector_store.texts) if hasattr(pipeline.vector_store, 'texts') else 'N/A'}")

        return True

    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_local_rag_no_ollama()
    if success:
        print("\n🎉 Test erfolgreich abgeschlossen!")
        print("\nNächste Schritte:")
        print("1. Installiere Ollama für vollständige lokale Nutzung")
        print("2. Oder nutze OpenAI API für Generation")
        print("3. Erweitere QA-Datensatz in data/evaluation/qa_pairs.json")
    else:
        print("\n❌ Test fehlgeschlagen!")
        sys.exit(1)