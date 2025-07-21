@echo off
REM ResearchRAG Lokales Setup-Skript für Windows

echo 🚀 ResearchRAG Lokales Setup
echo ==============================

REM 1. Dependencies installieren
echo 📦 Installiere Python-Dependencies...
pip install sentence-transformers chromadb faiss-cpu requests

REM 2. Ollama installieren (manuell erforderlich)
echo.
echo 🦙 Ollama Setup erforderlich:
echo    1. Gehe zu https://ollama.ai/download
echo    2. Lade Ollama für Windows herunter
echo    3. Installiere und starte Ollama
echo    4. Führe dann aus: ollama pull llama3.2
echo.

echo ✅ Python-Dependencies installiert!
echo.
echo 🧪 Nach Ollama-Setup teste das System mit:
echo    python test_local_rag.py
echo.
pause