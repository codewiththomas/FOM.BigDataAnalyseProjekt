#!/usr/bin/env python3
"""
Test Runner für das RAG-System.

Führt alle Tests aus und behebt Import-Probleme durch korrekte Pfad-Konfiguration.
"""

import os
import sys
import subprocess
from pathlib import Path

# Projektverzeichnis ermitteln
PROJECT_ROOT = Path(__file__).parent.absolute()
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"

# Pfade zu sys.path hinzufügen
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Umgebungsvariable für Python-Pfad setzen
os.environ["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{PROJECT_ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

def run_tests():
    """Führt alle Tests aus."""
    print("=" * 60)
    print("RAG-System Test Runner")
    print("=" * 60)

    # Test-Verzeichnis prüfen
    if not TESTS_DIR.exists():
        print(f"❌ Test-Verzeichnis nicht gefunden: {TESTS_DIR}")
        return False

    # Verfügbare Testdateien finden
    test_files = list(TESTS_DIR.glob("test_*.py"))

    if not test_files:
        print("❌ Keine Testdateien gefunden")
        return False

    print(f"📁 Projektverzeichnis: {PROJECT_ROOT}")
    print(f"📁 Src-Verzeichnis: {SRC_DIR}")
    print(f"📁 Test-Verzeichnis: {TESTS_DIR}")
    print(f"🧪 Gefundene Testdateien: {len(test_files)}")

    for test_file in test_files:
        print(f"   - {test_file.name}")

    print("\n" + "=" * 60)
    print("Starte Tests...")
    print("=" * 60)

    # Tests ausführen
    try:
        # Arbeitsverzeichnis wechseln
        os.chdir(PROJECT_ROOT)

        # pytest mit korrekten Parametern ausführen
        cmd = [
            sys.executable, "-m", "pytest",
            str(TESTS_DIR),
            "-v",
            "--tb=short",
            "--disable-warnings",
            "-x"  # Stoppe bei erstem Fehler
        ]

        print(f"🚀 Ausführung: {' '.join(cmd)}")
        print(f"📂 Arbeitsverzeichnis: {os.getcwd()}")
        print(f"🐍 Python-Pfad: {sys.executable}")
        print(f"📦 PYTHONPATH: {os.environ.get('PYTHONPATH', 'Nicht gesetzt')}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Ausgabe anzeigen
        if result.stdout:
            print("\n📋 Test-Ausgabe:")
            print(result.stdout)

        if result.stderr:
            print("\n⚠️  Fehler-Ausgabe:")
            print(result.stderr)

        # Ergebnis auswerten
        if result.returncode == 0:
            print("\n✅ Alle Tests erfolgreich!")
            return True
        else:
            print(f"\n❌ Tests fehlgeschlagen (Exit Code: {result.returncode})")
            return False

    except Exception as e:
        print(f"❌ Fehler beim Ausführen der Tests: {e}")
        return False

def run_specific_test(test_name: str):
    """Führt einen spezifischen Test aus."""
    print(f"🧪 Führe spezifischen Test aus: {test_name}")

    try:
        os.chdir(PROJECT_ROOT)

        cmd = [
            sys.executable, "-m", "pytest",
            f"{TESTS_DIR}/{test_name}",
            "-v",
            "--tb=short",
            "--disable-warnings"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print(result.stderr)

        return result.returncode == 0

    except Exception as e:
        print(f"❌ Fehler beim Ausführen des Tests: {e}")
        return False

def check_dependencies():
    """Prüft verfügbare Abhängigkeiten."""
    print("\n🔍 Prüfe verfügbare Abhängigkeiten...")

    dependencies = {
        "numpy": "✅ Erforderlich",
        "pandas": "✅ Erforderlich",
        "scikit-learn": "✅ Erforderlich",
        "sentence-transformers": "⚠️  Optional",
        "chromadb": "⚠️  Optional",
        "faiss-cpu": "⚠️  Optional",
        "openai": "⚠️  Optional",
        "requests": "✅ Erforderlich für Ollama"
    }

    for dep, status in dependencies.items():
        try:
            __import__(dep.replace("-", "_"))
            print(f"   ✅ {dep} - {status}")
        except ImportError:
            print(f"   ❌ {dep} - {status}")

    # Ollama-Verfügbarkeit prüfen
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=1)
        if response.status_code == 200:
            print("   ✅ Ollama Service - Verfügbar")
        else:
            print("   ❌ Ollama Service - Nicht verfügbar")
    except:
        print("   ❌ Ollama Service - Nicht verfügbar")

def run_quick_test():
    """Führt einen schnellen Test der Grundfunktionalität aus."""
    print("\n🚀 Schneller Funktionalitätstest...")

    try:
        # Basis-Imports testen
        from components.chunkers import LineChunker
        from components.embeddings import SentenceTransformerEmbedding
        from components.vector_stores import InMemoryVectorStore
        from evaluations import RetrievalEvaluator, GenerationEvaluator

        print("   ✅ Basis-Imports erfolgreich")

        # Einfache Komponenten-Tests
        chunker = LineChunker(chunk_size=100)
        chunks = chunker.chunk_text("Das ist ein Test. Das ist ein weiterer Test.")
        print(f"   ✅ Chunker: {len(chunks)} Chunks erstellt")

        # Embedding-Test (wenn verfügbar)
        try:
            embedding = SentenceTransformerEmbedding()
            if embedding.available:
                emb = embedding.embed_query("Test")
                print(f"   ✅ Embedding: Dimension {len(emb)}")
            else:
                print("   ⚠️  Embedding: Nicht verfügbar")
        except:
            print("   ⚠️  Embedding: Fehler")

        # Vector Store Test
        vector_store = InMemoryVectorStore()
        print("   ✅ Vector Store: Initialisiert")

        # Evaluator Test
        evaluator = RetrievalEvaluator()
        print("   ✅ Evaluator: Initialisiert")

        print("\n✅ Schnelltest erfolgreich!")
        return True

    except Exception as e:
        print(f"\n❌ Schnelltest fehlgeschlagen: {e}")
        return False

def main():
    """Hauptfunktion."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            check_dependencies()
            run_quick_test()
        elif sys.argv[1] == "deps":
            check_dependencies()
        elif sys.argv[1].startswith("test_"):
            run_specific_test(sys.argv[1])
        else:
            print("Verfügbare Optionen:")
            print("  python run_tests.py        - Alle Tests ausführen")
            print("  python run_tests.py quick  - Schnelltest")
            print("  python run_tests.py deps   - Abhängigkeiten prüfen")
            print("  python run_tests.py test_components.py - Spezifischen Test ausführen")
    else:
        check_dependencies()
        run_quick_test()
        run_tests()

if __name__ == "__main__":
    main()