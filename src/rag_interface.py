#!/usr/bin/env python3
"""
DSGVO RAG Interface - Einfache Benutzeroberfläche für das RAG-System
"""

import gradio as gr
import sys
import os
from pathlib import Path

# Add the src/rag directory to Python path
sys.path.append(str(Path(__file__).parent / "rag"))

try:
    from factory import RAGFactory
    from dataset import DSGVODataset
    from pipeline import RAGPipeline
except ImportError as e:
    print(f"Fehler beim Importieren der RAG-Module: {e}")
    print("Stellen Sie sicher, dass Sie sich im richtigen Verzeichnis befinden.")
    sys.exit(1)


class RAGInterface:
    """Einfache Gradio-Oberfläche für das RAG-System"""

    def __init__(self):
        self.pipeline = None
        self.is_initialized = False

    def initialize_rag(self, config_path, dataset_path):
        """RAG-System initialisieren"""
        try:
            if not config_path or not dataset_path:
                return "❌ Bitte geben Sie beide Pfade an."

            if not os.path.exists(config_path):
                return "❌ Konfigurationsdatei nicht gefunden."

            if not os.path.exists(dataset_path):
                return "❌ Datensatz nicht gefunden."

            # Factory erstellen
            factory = RAGFactory(config_path)
            pipeline = factory.create_pipeline()

            # Datensatz laden und indexieren
            dataset = DSGVODataset(dataset_path)
            pipeline.index_documents(dataset.documents)

            # Erfolg
            self.pipeline = pipeline
            self.is_initialized = True

            return (
                f"✅ RAG-System erfolgreich initialisiert!\n\n"
                f"📊 Dokumente verarbeitet: {len(dataset.documents)}\n"
                f"🔧 Konfiguration: {os.path.basename(config_path)}\n"
                f"📁 Datensatz: {os.path.basename(dataset_path)}\n\n"
                f"Sie können jetzt Fragen zur DSGVO stellen!"
            )

        except Exception as e:
            return f"❌ Fehler bei der Initialisierung: {str(e)}"

    def ask_question(self, question):
        """Frage an das RAG-System stellen"""
        if not self.is_initialized:
            return "❌ Bitte initialisieren Sie zuerst das RAG-System."

        if not question.strip():
            return "❌ Bitte geben Sie eine Frage ein."

        try:
            # Antwort generieren
            result = self.pipeline.query(question.strip())

            # Antwort formatieren
            response = f"🤖 **Antwort:**\n{result.response}\n\n"
            response += f"📊 **Metadaten:**\n"
            response += f"• Antwortzeit: {result.metadata.get('query_time', 'N/A'):.2f}s\n"
            response += f"• Relevante Textabschnitte: {result.metadata.get('chunks_retrieved', 'N/A')}\n"
            response += f"• Kontextlänge: {result.metadata.get('context_length', 'N/A')} Zeichen"

            return response

        except Exception as e:
            return f"❌ Fehler bei der Antwortgenerierung: {str(e)}"

    def create_interface(self):
        """Gradio-Interface erstellen"""

        with gr.Blocks(
            title="DSGVO RAG Chat Interface",
            theme=gr.themes.Soft()
        ) as interface:

            gr.Markdown("# Research RAG Chat Interface")
            gr.Markdown("Ein intelligentes Frage-Antwort-System basierend auf der DSGVO")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## ⚙️ Konfiguration")

                    config_path = gr.Textbox(
                        label="Konfigurationsdatei (YAML)",
                        value="../configs/003_llama32b.yaml",
                        placeholder="Pfad zur YAML-Konfigurationsdatei"
                    )

                    dataset_path = gr.Textbox(
                        label="Datensatz (JSONL)",
                        value="../data/output/dsgvo_crawled_2025-08-20_1824.jsonl",
                        placeholder="Pfad zur JSONL-Datensatzdatei"
                    )

                    init_button = gr.Button(
                        "🚀 RAG-System initialisieren",
                        variant="primary",
                        size="lg"
                    )

                    status_output = gr.Textbox(
                        label="Status",
                        value="Bereit zur Initialisierung",
                        interactive=False,
                        lines=5
                    )

                with gr.Column(scale=2):
                    gr.Markdown("## 💬 Chat")

                    chat_output = gr.Markdown(
                        value="👋 Willkommen! Initialisieren Sie das RAG-System, um Fragen zur DSGVO zu stellen.",
                        height=400
                    )

                    with gr.Row():
                        question_input = gr.Textbox(
                            label="Ihre Frage",
                            placeholder="Stellen Sie eine Frage zur DSGVO...",
                            scale=4
                        )
                        ask_button = gr.Button("❓ Frage stellen", variant="secondary", scale=1)

                    # gr.Markdown("**Beispielfragen:**")
                    # gr.Markdown("• Was sind die Grundprinzipien der DSGVO?")
                    # gr.Markdown("• Welche Rechte haben Betroffene?")
                    # gr.Markdown("• Was ist eine Datenschutz-Folgenabschätzung?")

            # Event-Handler
            init_button.click(
                fn=self.initialize_rag,
                inputs=[config_path, dataset_path],
                outputs=[status_output]
            )

            ask_button.click(
                fn=self.ask_question,
                inputs=[question_input],
                outputs=[chat_output]
            )

            question_input.submit(
                fn=self.ask_question,
                inputs=[question_input],
                outputs=[chat_output]
            )

        return interface


def main():
    """Hauptfunktion"""
    print("🚀 Starte DSGVO RAG Interface...")

    # Interface erstellen
    rag_interface = RAGInterface()
    interface = rag_interface.create_interface()

    # Interface starten
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )


if __name__ == "__main__":
    main()
