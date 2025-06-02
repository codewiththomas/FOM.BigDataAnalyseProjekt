"""
Implementiert eine einfache Web-UI für das RAG-System.
"""

from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO
import os
import json
from typing import Optional

from ..components.chain.base import RAGChain


class ChatUI:
    """
    Implementiert eine einfache Web-UI für das RAG-System.
    """

    def __init__(self, chain: RAGChain, port: int = 5000, debug: bool = False):
        """
        Initialisiert die ChatUI.

        Args:
            chain: Die zu verwendende RAG-Chain
            port: Der Port, auf dem die Web-UI laufen soll
            debug: Ob Debug-Modus aktiviert sein soll
        """
        self.chain = chain
        self.port = port
        self.debug = debug

        # Flask-App initialisieren
        self.app = Flask(
            __name__,
            template_folder=os.path.join(os.path.dirname(__file__), "templates"),
            static_folder=os.path.join(os.path.dirname(__file__), "static")
        )

        # SocketIO initialisieren
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Routen registrieren
        self._register_routes()

    def _register_routes(self):
        """Registriert die Routen für die Web-UI."""

        @self.app.route("/")
        def index():
            """Startseite der Web-UI."""
            return render_template("index.html")

        @self.app.route("/api/chat", methods=["POST"])
        def chat():
            """
            API-Endpunkt für Chat-Anfragen.

            Returns:
                JSON-Antwort mit generiertem Text
            """
            data = request.get_json()
            query = data.get("query", "")

            if not query:
                return jsonify({"error": "Keine Anfrage gesendet"}), 400

            try:
                response = self.chain.run(query)
                return jsonify({"response": response})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/api/chat/stream", methods=["POST"])
        def chat_stream():
            """
            API-Endpunkt für Chat-Anfragen mit Streaming.

            Returns:
                Streaming-Response mit generiertem Text
            """
            data = request.get_json()
            query = data.get("query", "")

            if not query:
                return jsonify({"error": "Keine Anfrage gesendet"}), 400

            def generate():
                try:
                    for token in self.chain.run_stream(query):
                        chunk = json.dumps({"token": token}) + "\n"
                        yield chunk
                except Exception as e:
                    yield json.dumps({"error": str(e)}) + "\n"

            return Response(generate(), mimetype="text/event-stream")

        @self.socketio.on("chat")
        def handle_chat(data):
            """
            Socket.IO-Event für Chat-Anfragen.

            Args:
                data: Die Anfragedaten mit dem Query
            """
            query = data.get("query", "")

            if not query:
                self.socketio.emit("error", {"message": "Keine Anfrage gesendet"})
                return

            try:
                # Streaming-Antwort generieren
                for token in self.chain.run_stream(query):
                    self.socketio.emit("token", {"token": token})

                # Abschluss-Event senden
                self.socketio.emit("done")
            except Exception as e:
                self.socketio.emit("error", {"message": str(e)})

    def run(self):
        """Startet die Web-UI."""
        self.socketio.run(self.app, port=self.port, debug=self.debug)