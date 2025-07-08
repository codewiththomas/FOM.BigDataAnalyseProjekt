#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test-Skript für das RAG Notebook
"""

# 📦 Setup & Imports
import sys
import os
from pathlib import Path

# Pfad-Setup für src/
project_root = Path.cwd()
if project_root.name == 'src':
    project_root = project_root.parent
sys.path.insert(0, str(project_root / 'src'))

print("🔍 Teste Notebook-Funktionalität...")
print(f"📁 Projekt-Root: {project_root}")
print(f"📁 Aktueller Pfad: {Path.cwd()}")
print(f"🐍 Python-Pfad: {sys.path[:3]}")

try:
    # Core Imports
    from core.research_rag import ResearchRAG
    print("✅ ResearchRAG Import erfolgreich")
except ImportError as e:
    print(f"❌ ResearchRAG Import fehlgeschlagen: {e}")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    print("✅ dotenv Import erfolgreich")
except ImportError as e:
    print(f"⚠️ dotenv nicht verfügbar: {e}")
    load_dotenv = lambda x: None

# API Key laden
load_dotenv(project_root / '.env')
api_key = os.getenv('OPENAI_API_KEY', 'demo-key')

print('✅ Research RAG System bereit')
print(f'🔑 API Key: {"✅ Geladen" if api_key != "demo-key" else "⚠️ Demo-Modus"}')

# 🔧 RAG-Konfiguration (einfach anpassen)
config = {
    # Chunking-Strategie
    'chunker': 'line',  # 'line' oder 'recursive'
    'chunk_size': 1000,
    'chunk_overlap': 200,

    # Embedding-Modell
    'embedding': 'sentence_transformers',  # 'sentence_transformers' oder 'openai'

    # Vector Store
    'vector_store': 'in_memory',  # 'in_memory' oder 'chroma'

    # Language Model
    'llm': 'openai',  # 'openai' oder 'local'
    'api_key': api_key,

    # Evaluation
    'num_test_questions': 3,  # Reduziert für Test
    'top_k': 5  # Top-K für Retrieval
}

print('\n⚙️ Konfiguration:')
for key, value in config.items():
    if key != 'api_key':
        print(f'   {key}: {value}')
print('✅ Konfiguration bereit')

# 🚀 RAG-System initialisieren
try:
    print('\n🔄 Initialisiere RAG-System...')
    rag = ResearchRAG(config)
    print('✅ RAG-System erfolgreich initialisiert')

    # Kurzer Test ohne vollständige Evaluation
    print('\n🧪 Teste Basis-Funktionalität...')

    # Prüfe ob Daten geladen werden können
    if hasattr(rag, 'data_loader'):
        print('✅ DataLoader verfügbar')

    # Prüfe Komponenten
    if hasattr(rag, 'chunker'):
        print('✅ Chunker verfügbar')
    if hasattr(rag, 'embedding'):
        print('✅ Embedding verfügbar')
    if hasattr(rag, 'vector_store'):
        print('✅ Vector Store verfügbar')
    if hasattr(rag, 'llm'):
        print('✅ Language Model verfügbar')

    print('\n🎉 Notebook-Test erfolgreich!')
    print('📝 Das Notebook sollte funktionieren.')

except Exception as e:
    print(f'❌ Fehler beim Initialisieren: {e}')
    import traceback
    traceback.print_exc()