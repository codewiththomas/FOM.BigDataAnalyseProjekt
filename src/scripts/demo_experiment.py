#!/usr/bin/env python3
"""
Demo Baseline-Experiment für RAG-System
Zeigt die Evaluationsmetriken ohne externe API-Aufrufe
"""

import os
import sys
import time
import json
from datetime import datetime

# Pfade konfigurieren
sys.path.append('.')
sys.path.append('src')

def calculate_precision_recall_f1(retrieved_docs, relevant_docs):
    """Berechnet Precision, Recall und F1-Score"""
    if not retrieved_docs or not relevant_docs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)

    tp = len(retrieved_set.intersection(relevant_set))
    fp = len(retrieved_set - relevant_set)
    fn = len(relevant_set - retrieved_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}

def calculate_context_relevance(context, question):
    """Berechnet Context Relevance Score"""
    if not context or not question:
        return 0.0

    question_words = set(question.lower().split())
    total_relevance = 0.0

    for ctx in context:
        context_words = set(ctx.lower().split())
        overlap = len(question_words.intersection(context_words))
        relevance = overlap / len(question_words) if question_words else 0.0
        total_relevance += relevance

    return total_relevance / len(context)

def calculate_answer_relevance(answer, question):
    """Berechnet Answer Relevance Score"""
    if not answer or not question:
        return 0.0

    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())

    overlap = len(question_words.intersection(answer_words))
    relevance = overlap / len(question_words) if question_words else 0.0

    return min(relevance, 1.0)

def simulate_rag_system_results():
    """Simuliert RAG-System Ergebnisse für Demo-Zwecke"""

    # Simulierte DSGVO-Dokumente
    documents = [
        "Artikel 1 DSGVO: Gegenstand und Ziele dieser Verordnung",
        "Artikel 2 DSGVO: Sachlicher Anwendungsbereich",
        "Artikel 3 DSGVO: Räumlicher Anwendungsbereich",
        "Artikel 4 DSGVO: Begriffsbestimmungen - personenbezogene Daten",
        "Artikel 5 DSGVO: Grundsätze für die Verarbeitung personenbezogener Daten",
        "Artikel 6 DSGVO: Rechtmäßigkeit der Verarbeitung",
        "Artikel 7 DSGVO: Bedingungen für die Einwilligung",
        "Artikel 8 DSGVO: Bedingungen für die Einwilligung eines Kindes",
        "Artikel 9 DSGVO: Verarbeitung besonderer Kategorien personenbezogener Daten",
        "Artikel 10 DSGVO: Verarbeitung von Daten über strafrechtliche Verurteilungen",
        "Artikel 11 DSGVO: Verarbeitung ohne Identifizierung der betroffenen Person",
        "Artikel 12 DSGVO: Transparente Information und Kommunikation",
        "Artikel 13 DSGVO: Informationspflicht bei Erhebung von Daten",
        "Artikel 14 DSGVO: Informationspflicht bei Datenerhebung von Dritten",
        "Artikel 15 DSGVO: Auskunftsrecht der betroffenen Person",
        "Artikel 16 DSGVO: Recht auf Berichtigung",
        "Artikel 17 DSGVO: Recht auf Löschung (Recht auf Vergessenwerden)",
        "Artikel 18 DSGVO: Recht auf Einschränkung der Verarbeitung",
        "Artikel 19 DSGVO: Mitteilungspflicht bezüglich Berichtigung oder Löschung",
        "Artikel 20 DSGVO: Recht auf Datenübertragbarkeit"
    ]

    # Test-Fragen
    questions = [
        "Was sind personenbezogene Daten nach DSGVO?",
        "Welche Rechte haben betroffene Personen?",
        "Wann ist eine Einwilligung erforderlich?",
        "Was ist das Recht auf Vergessenwerden?",
        "Welche Grundsätze gelten für die Datenverarbeitung?"
    ]

    # Simulierte Antworten (würden normalerweise von LLM generiert)
    answers = [
        "Personenbezogene Daten sind alle Informationen, die sich auf eine identifizierte oder identifizierbare natürliche Person beziehen.",
        "Betroffene Personen haben das Recht auf Auskunft, Berichtigung, Löschung, Einschränkung der Verarbeitung und Datenübertragbarkeit.",
        "Eine Einwilligung ist erforderlich, wenn keine andere Rechtsgrundlage für die Verarbeitung vorliegt.",
        "Das Recht auf Vergessenwerden ermöglicht es Personen, die Löschung ihrer personenbezogenen Daten zu verlangen.",
        "Die Grundsätze umfassen Rechtmäßigkeit, Transparenz, Zweckbindung, Datenminimierung und Speicherbegrenzung."
    ]

    return documents, questions, answers

def main():
    print("🚀 Demo Baseline-Experiment für RAG-System")
    print("📊 Evaluiert: Precision@5, Recall@5, F1-Score, RAGAS, Inferenzgeschwindigkeit")
    print("=" * 80)

    # Simulierte Daten laden
    print("\n📄 Lade simulierte DSGVO-Daten...")
    documents, questions, answers = simulate_rag_system_results()

    print(f"✅ {len(documents)} Dokumente geladen")
    print(f"✅ {len(questions)} Test-Fragen erstellt")

    # Simuliere Dokumentverarbeitung
    print("\n📚 Simuliere Dokumentverarbeitung...")
    processing_start = time.time()
    time.sleep(0.5)  # Simuliere Verarbeitungszeit
    processing_time = time.time() - processing_start
    print(f"✅ Dokumente verarbeitet in {processing_time:.2f}s")

    # Simuliere Query-Ausführung
    print("\n💬 Simuliere Query-Ausführung...")
    query_times = []

    for i, question in enumerate(questions):
        print(f"   {i+1}. {question}")

        # Simuliere Query-Zeit
        query_start = time.time()
        time.sleep(0.1)  # Simuliere Antwortzeit
        query_time = time.time() - query_start
        query_times.append(query_time)

        print(f"      ⏱️  {query_time:.3f}s")
        print(f"      💡 {answers[i][:80]}...")

    # Simuliere Retrieval-Ergebnisse
    print("\n📊 Berechne Metriken...")

    # Für jede Frage simulieren wir retrieved und relevant documents
    retrieved_docs_per_question = []
    relevant_docs_per_question = []

    for i, question in enumerate(questions):
        # Simuliere: Top 5 retrieved documents
        retrieved = documents[i:i+5] if i+5 <= len(documents) else documents[:5]

        # Simuliere: Relevante Dokumente basierend auf Schlüsselwörtern
        relevant = []
        question_lower = question.lower()

        for doc in documents:
            # Einfache Relevanz-Simulation
            if any(word in doc.lower() for word in question_lower.split() if len(word) > 3):
                relevant.append(doc)

        # Mindestens 3 relevante Dokumente
        if len(relevant) < 3:
            relevant.extend(documents[:3])

        retrieved_docs_per_question.append(retrieved)
        relevant_docs_per_question.append(relevant[:5])

    # Retrieval-Metriken berechnen
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for retrieved, relevant in zip(retrieved_docs_per_question, relevant_docs_per_question):
        metrics = calculate_precision_recall_f1(retrieved, relevant)
        all_precisions.append(metrics["precision"])
        all_recalls.append(metrics["recall"])
        all_f1s.append(metrics["f1"])

    # RAGAS-Metriken berechnen
    context_relevances = []
    answer_relevances = []

    for question, answer, context in zip(questions, answers, retrieved_docs_per_question):
        context_rel = calculate_context_relevance(context, question)
        answer_rel = calculate_answer_relevance(answer, question)

        context_relevances.append(context_rel)
        answer_relevances.append(answer_rel)

    # Durchschnittswerte und Standardabweichungen
    def calc_stats(values):
        if not values:
            return 0.0, 0.0
        mean = sum(values) / len(values)
        if len(values) == 1:
            return mean, 0.0
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        std = variance ** 0.5
        return mean, std

    avg_precision, std_precision = calc_stats(all_precisions)
    avg_recall, std_recall = calc_stats(all_recalls)
    avg_f1, std_f1 = calc_stats(all_f1s)
    avg_context_rel, std_context_rel = calc_stats(context_relevances)
    avg_answer_rel, std_answer_rel = calc_stats(answer_relevances)

    # Inferenzgeschwindigkeit
    total_query_time = sum(query_times)
    avg_query_time = total_query_time / len(questions)
    queries_per_second = len(questions) / total_query_time

    # Ergebnisse strukturieren
    results = {
        "timestamp": datetime.now().isoformat(),
        "configuration": "Baseline: Line + SentenceTransformers + InMemory + OpenAI",
        "data_stats": {
            "num_documents": len(documents),
            "num_questions": len(questions)
        },
        "retrieval_metrics": {
            "precision@5": avg_precision,
            "recall@5": avg_recall,
            "f1@5": avg_f1,
            "std_precision@5": std_precision,
            "std_recall@5": std_recall,
            "std_f1@5": std_f1
        },
        "ragas_metrics": {
            "context_relevance": avg_context_rel,
            "answer_relevance": avg_answer_rel,
            "std_context_relevance": std_context_rel,
            "std_answer_relevance": std_answer_rel
        },
        "speed_metrics": {
            "document_processing_time": processing_time,
            "total_query_time": total_query_time,
            "avg_query_time": avg_query_time,
            "queries_per_second": queries_per_second
        },
        "sample_qa": [
            {"question": q, "answer": a}
            for q, a in zip(questions, answers)
        ]
    }

    # Ergebnisse anzeigen
    print("\n" + "="*80)
    print("📊 BASELINE-EXPERIMENT ERGEBNISSE")
    print("="*80)

    print(f"⏰ Zeitstempel: {results['timestamp']}")
    print(f"📄 Dokumente: {results['data_stats']['num_documents']}")
    print(f"❓ Fragen: {results['data_stats']['num_questions']}")

    print(f"\n🎯 RETRIEVAL-METRIKEN:")
    ret_metrics = results['retrieval_metrics']
    print(f"   Precision@5: {ret_metrics['precision@5']:.3f} (±{ret_metrics['std_precision@5']:.3f})")
    print(f"   Recall@5:    {ret_metrics['recall@5']:.3f} (±{ret_metrics['std_recall@5']:.3f})")
    print(f"   F1@5:        {ret_metrics['f1@5']:.3f} (±{ret_metrics['std_f1@5']:.3f})")

    print(f"\n📝 RAGAS-METRIKEN:")
    ragas_metrics = results['ragas_metrics']
    print(f"   Context Relevance: {ragas_metrics['context_relevance']:.3f} (±{ragas_metrics['std_context_relevance']:.3f})")
    print(f"   Answer Relevance:  {ragas_metrics['answer_relevance']:.3f} (±{ragas_metrics['std_answer_relevance']:.3f})")

    print(f"\n⚡ INFERENZGESCHWINDIGKEIT:")
    speed_metrics = results['speed_metrics']
    print(f"   Dokumentverarbeitung: {speed_metrics['document_processing_time']:.2f}s")
    print(f"   Queries/Sekunde:      {speed_metrics['queries_per_second']:.2f}")
    print(f"   Ø Zeit/Query:         {speed_metrics['avg_query_time']:.3f}s")

    print(f"\n💬 BEISPIEL-ANTWORTEN:")
    for i, qa in enumerate(results['sample_qa']):
        print(f"   {i+1}. Q: {qa['question']}")
        print(f"      A: {qa['answer']}")
        print()

    # Detaillierte Metrik-Aufschlüsselung
    print(f"\n📈 DETAILLIERTE METRIK-AUFSCHLÜSSELUNG:")
    print(f"   Pro Frage:")
    for i, (q, p, r, f1, cr, ar) in enumerate(zip(
        questions, all_precisions, all_recalls, all_f1s,
        context_relevances, answer_relevances
    )):
        print(f"   {i+1}. {q[:50]}...")
        print(f"      Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}")
        print(f"      Context Rel: {cr:.3f}, Answer Rel: {ar:.3f}")
        print()

    # Ergebnisse speichern
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"results/baseline_experiment_{timestamp}.json"

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"💾 Ergebnisse gespeichert: {filepath}")
    print("\n✅ Baseline-Experiment erfolgreich abgeschlossen!")

if __name__ == "__main__":
    main()