#!/usr/bin/env python3
"""
Test-Skript für lokales ResearchRAG-System
Testet die vollständig lokale Pipeline ohne API-Abhängigkeiten
"""

import sys
import os
import json
from pathlib import Path

# Pfad zum src-Verzeichnis hinzufügen
sys.path.insert(0, 'src')

def test_local_rag():
    print("🚀 Teste lokales ResearchRAG-System...")

    try:
        # Imports
        from config.pipeline_configs import get_local_config
        from core.rag_pipeline import RAGPipeline

        print("✅ Module erfolgreich importiert")

        # Lokale Konfiguration laden
        config = get_local_config()
        print(f"✅ Konfiguration geladen: {config.get_component_types()}")

        # Pipeline erstellen
        print("🔧 Erstelle Pipeline...")
        pipeline = RAGPipeline(config)
        print("✅ Pipeline erstellt!")

        # DSGVO-Dokument laden
        dsgvo_file = "data/raw/dsgvo.txt"
        if not Path(dsgvo_file).exists():
            print(f"❌ DSGVO-Datei nicht gefunden: {dsgvo_file}")
            return False

        print("📄 Lade DSGVO-Dokument...")
        documents = pipeline.load_documents_from_file(dsgvo_file)
        print(f"✅ Dokument geladen: {len(documents[0]):,} Zeichen")

        # Indexieren
        print("🔍 Indexiere Dokument...")
        stats = pipeline.index_documents(documents)
        print("✅ Indexierung abgeschlossen!")
        print(f"   - {stats['total_chunks']} Chunks erstellt")
        print(f"   - {stats['embedding_dimension']} Embedding-Dimensionen")
        print(f"   - {stats['indexing_time']:.2f}s Indexierungszeit")

        # Test-Query
        test_question = "Was ist die maximale Geldbuße nach Art. 83 DSGVO?"
        print(f"\n❓ Test-Query: {test_question}")

        result = pipeline.query(test_question, return_context=True)
        print(f"✅ Antwort erhalten in {result['query_time']:.2f}s")
        print(f"\n📝 Antwort:\n{result['answer']}")

        # Kontext anzeigen
        contexts = result.get('retrieved_contexts', [])
        print(f"\n🔍 {len(contexts)} relevante Kontexte gefunden:")
        for i, ctx in enumerate(contexts[:2], 1):
            print(f"   {i}. {ctx['text'][:100]}...")

        return True

    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qa_evaluation():
    """Testet die QA-Evaluierung mit einigen Beispielfragen"""
    print("\n🧪 Teste QA-Evaluierung...")

    try:
        from config.pipeline_configs import get_local_config
        from core.rag_pipeline import RAGPipeline

        # Pipeline erstellen (falls nicht schon vorhanden)
        config = get_local_config()
        pipeline = RAGPipeline(config)

        # Dokument laden falls nötig
        if not pipeline.is_indexed:
            documents = pipeline.load_documents_from_file("data/raw/dsgvo.txt")
            pipeline.index_documents(documents, show_progress=False)

        # QA-Pairs laden
        qa_file = "data/evaluation/qa_pairs.json"
        if not Path(qa_file).exists():
            print(f"❌ QA-Datei nicht gefunden: {qa_file}")
            return False

        with open(qa_file, "r", encoding="utf-8") as f:
            qa_data = json.load(f)

        print(f"✅ {qa_data['metadata']['total_questions']} QA-Pairs geladen")

        # Erste 3 Fragen testen
        print("\n📋 Teste erste 3 Fragen:")
        for i, qa in enumerate(qa_data["questions"][:3], 1):
            print(f"\n--- Frage {i} ---")
            print(f"❓ {qa['question']}")

            result = pipeline.query(qa["question"])
            print(f"🤖 Antwort: {result['answer']}")
            print(f"✅ Gold: {qa['gold_answer']}")
            print(f"⏱️  Zeit: {result['query_time']:.2f}s")
            print("-" * 50)

        return True

    except Exception as e:
        print(f"❌ Fehler bei QA-Evaluierung: {e}")
        return False

def check_dependencies():
    """Prüft ob alle erforderlichen Dependencies verfügbar sind"""
    print("🔍 Prüfe Dependencies...")

    deps = {
        "sentence-transformers": "sentence_transformers",
        "requests": "requests",
        "numpy": "numpy",
        "scikit-learn": "sklearn"
    }

    missing = []
    for name, module in deps.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - FEHLT!")
            missing.append(name)

    if missing:
        print(f"\n⚠️  Fehlende Dependencies: {', '.join(missing)}")
        print("Installiere mit: pip install " + " ".join(missing))
        return False

    return True

def check_ollama():
    """Prüft ob Ollama verfügbar ist"""
    print("\n🦙 Prüfe Ollama...")

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            model_names = [m['name'] for m in models.get('models', [])]
            print(f"✅ Ollama läuft - {len(model_names)} Modelle verfügbar")

            if 'llama3.2' in model_names:
                print("✅ llama3.2 verfügbar")
                return True
            else:
                print("❌ llama3.2 nicht gefunden")
                print("Lade mit: ollama pull llama3.2")
                return False
        else:
            print("❌ Ollama antwortet nicht korrekt")
            return False
    except Exception as e:
        print(f"❌ Ollama nicht erreichbar: {e}")
        print("Starte Ollama mit: ollama serve")
        return False

if __name__ == "__main__":
    print("🔧 ResearchRAG Lokaler Test")
    print("=" * 50)

    # Dependencies prüfen
    if not check_dependencies():
        print("\n❌ Dependencies fehlen. Installiere sie zuerst.")
        sys.exit(1)

    # Ollama prüfen
    if not check_ollama():
        print("\n❌ Ollama nicht verfügbar. Setup erforderlich.")
        sys.exit(1)

    # Haupttest
    if test_local_rag():
        print("\n🎉 Basis-Test erfolgreich!")

        # QA-Test
        if test_qa_evaluation():
            print("\n🎉 QA-Test erfolgreich!")
            print("\n✅ ResearchRAG ist vollständig einsatzbereit!")
        else:
            print("\n⚠️  QA-Test fehlgeschlagen, aber Basis funktioniert")
    else:
        print("\n❌ Test fehlgeschlagen")
        sys.exit(1)