# Start

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Getting Started

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Projektstruktur

    docs/                   # Enthält die Dokumentationsdateien.
        index.md
        ...
    src/rag/                # Enthält den eigentlichen Quellcode.
        components/         # Enthält die Komponenten des Projekts.
            chain/          # Enthält die Klassen für die RAG-Pipeline.
            data_sources    # Enthält die Klassen für die Datenquellen.
            embeddings/     # Enthält die Klassen für die Embeddings.
            llm/            # Enthält die Klassen für das Language Model.
            retrievers/     # Enthält die Klassen für die Retriever.
            vector_stores/  # Enthält die Klassen für die Vector Stores.
        web/
            templates/
        app.py
        config.py
    tests/                  # Enthält die Tests für den Quellcode.
    mkdocs.yml              # Konfigurationsdatei für Dokumentationsprojekt.
