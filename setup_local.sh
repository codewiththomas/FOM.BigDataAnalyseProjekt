#!/bin/bash
# ResearchRAG Lokales Setup-Skript

echo "🚀 ResearchRAG Lokales Setup"
echo "=============================="

# 1. Dependencies installieren
echo "📦 Installiere Python-Dependencies..."
pip install sentence-transformers chromadb faiss-cpu requests

# 2. Ollama installieren (falls nicht vorhanden)
if ! command -v ollama &> /dev/null; then
    echo "🦙 Installiere Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
else
    echo "✅ Ollama bereits installiert"
fi

# 3. Ollama starten (im Hintergrund)
echo "🔧 Starte Ollama-Service..."
ollama serve &
OLLAMA_PID=$!
sleep 5

# 4. Llama3.2 Modell laden
echo "📥 Lade llama3.2 Modell..."
ollama pull llama3.2

echo "✅ Setup abgeschlossen!"
echo ""
echo "🧪 Teste das System mit:"
echo "   python test_local_rag.py"
echo ""
echo "🛑 Stoppe Ollama mit:"
echo "   kill $OLLAMA_PID"