"""
Hauptmodul für das RAG-System.
Initialisiert alle Komponenten und startet die Web-UI.
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from dataclasses import asdict

from src.rag.config import config
from src.rag.components.data_sources.document_loader import TextFileLoader, DirectoryLoader
from src.rag.components.data_sources.text_splitter import RecursiveCharacterTextSplitter
from src.rag.components.embeddings.openai_embeddings import OpenAIEmbedder
from src.rag.components.vector_stores.factory import create_vector_store
from src.rag.components.vector_stores.base import Document
from src.rag.components.data_sources.base import Document as SourceDocument
from src.rag.components.llm.openai_llm import OpenAILanguageModel
from src.rag.components.retrieval.simple_retriever import SimpleRetriever
from src.rag.components.chain.rag_chain import SimpleRAGChain
from src.rag.web.chat_ui import ChatUI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue


def initialize_rag_system():
    """
    Initialisiert alle Komponenten des RAG-Systems.

    Returns:
        Eine initialisierte RAG-Chain
    """
    # Konfiguration validieren
    config.validate()

    # Pfade definieren
    data_dir = Path(config.data_dir)
    raw_data_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    embeddings_file = processed_dir / "dsgvo_embeddings.pkl"

    # Verzeichnisse erstellen, falls sie nicht existieren
    processed_dir.mkdir(exist_ok=True, parents=True)

    # Vektor Store initialisieren
    print("Initialisiere Vector Store...")
    vector_store = create_vector_store()

    # Embedder initialisieren (für Queries, auch wenn wir gespeicherte Embeddings haben)
    embedder = OpenAIEmbedder()

    # Prüfen, ob gespeicherte Embeddings existieren
    if embeddings_file.exists():
        print(f"Lade gespeicherte Embeddings aus {embeddings_file}...")
        try:
            with open(embeddings_file, 'rb') as f:
                data = pickle.load(f)
                chunks_data = data['chunks']
                embeddings = data['embeddings']

            print(f"Geladen: {len(chunks_data)} Chunks mit Embeddings")
            
            # Erstelle Vector Store Documents aus den geladenen Daten
            vector_documents = []
            for chunk_dict, embedding in zip(chunks_data, embeddings):
                # Erstelle ein Source Document aus den gespeicherten Daten
                source_doc = SourceDocument(
                    content=chunk_dict['content'],
                    metadata=chunk_dict['metadata'],
                    id=chunk_dict.get('id')
                )
                
                # Erstelle ein Vector Store Document
                vector_doc = Document(
                    text=source_doc.content,
                    embedding=np.array(embedding, dtype=np.float32),
                    metadata=source_doc.metadata.copy() if source_doc.metadata else {}
                )
                vector_documents.append(vector_doc)
            
            # Füge die Vector Store Documents zum Vector Store hinzu
            vector_store.add_documents(vector_documents)
            print(f"Dokumente in Qdrant gespeichert")
        except Exception as e:
            print(f"Fehler beim Laden der gespeicherten Embeddings: {e}")
            print("Erstelle neue Embeddings...")
            create_and_store_embeddings(raw_data_dir, embeddings_file, vector_store, embedder)
    else:
        print("Keine gespeicherten Embeddings gefunden.")
        create_and_store_embeddings(raw_data_dir, embeddings_file, vector_store, embedder)

    # Language Model initialisieren
    print("Initialisiere Language Model...")
    llm = OpenAILanguageModel()

    # Retriever initialisieren
    print("Initialisiere Retriever...")
    retriever = SimpleRetriever(
        vector_store=vector_store,
        embedder=embedder
    )

    # RAG-Chain initialisieren
    print("Initialisiere RAG-Chain...")
    chain = SimpleRAGChain(
        retriever=retriever,
        llm=llm
    )

    print("RAG-System erfolgreich initialisiert!")

    return chain


def create_and_store_embeddings(raw_data_dir, embeddings_file, vector_store, embedder):
    """
    Erstellt Embeddings für die DSGVO-Datei und speichert sie für zukünftige Nutzung.

    Args:
        raw_data_dir: Verzeichnis mit den Rohdaten
        embeddings_file: Pfad zur Datei, in der die Embeddings gespeichert werden sollen
        vector_store: Vector Store, in dem die Embeddings gespeichert werden sollen
        embedder: Embedder, mit dem die Embeddings erstellt werden sollen
    """
    # DSGVO-Datei laden
    dsgvo_path = raw_data_dir / "dsgvo.txt"

    if not dsgvo_path.exists():
        print(f"Fehler: DSGVO-Datei {dsgvo_path} existiert nicht.")
        print("Bitte stelle sicher, dass die DSGVO-Daten in der Datei data/raw/dsgvo.txt liegen.")
        sys.exit(1)

    # Dokumente aus DSGVO-Datei laden
    print("Lade DSGVO-Daten...")
    loader = TextFileLoader(str(dsgvo_path))
    documents = loader.load()
    print(f"Geladen: {len(documents)} Dokumente")

    # Text in Chunks aufteilen
    print("Teile Text in Chunks auf...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Erstellt: {len(chunks)} Chunks")

    # Embeddings erstellen und Vector Store Documents erzeugen
    print("Erstelle Embeddings...")
    print("Berechne Embeddings und speichere sie in Qdrant...")
    try:
        # Erstelle Vector Store Documents mit Embeddings
        vector_documents = embedder.embed_documents(chunks)
        
        # Speichere die Dokumente im Vector Store
        vector_store.add_documents(vector_documents)
        print("Dokumente erfolgreich in Qdrant gespeichert")

        # Embeddings für zukünftige Nutzung speichern
        print(f"Speichere Embeddings in {embeddings_file}...")
        with open(embeddings_file, 'wb') as f:
            # Konvertiere die Chunks in Dictionaries für die Speicherung
            chunks_data = [asdict(chunk) for chunk in chunks]
            # Speichere die Chunks und ihre Embeddings getrennt
            pickle.dump({
                'chunks': chunks_data,
                'embeddings': [doc.embedding.tolist() for doc in vector_documents]
            }, f)
    except Exception as e:
        print(f"Fehler beim Erstellen der Embeddings: {e}")
        sys.exit(1)


def main():
    """Hauptfunktion zum Starten des RAG-Systems."""
    print("Starte RAG-System...")

    # RAG-System initialisieren
    chain = initialize_rag_system()

    # Web-UI starten
    print(f"Starte Web-UI auf Port {config.web.port}...")
    ui = ChatUI(
        chain=chain,
        port=config.web.port,
        debug=config.web.debug
    )

    # Web-UI starten
    ui.run()


# Nur im Hauptprozess oder beim tatsächlichen direkten Aufruf ausführen
# Verhindert doppelte Initialisierung beim Flask-Debug-Reloader
if __name__ == "__main__" or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    main()