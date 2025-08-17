from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def group_documents_by_article(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Gruppiere DSGVO-Dokumente nach Artikel (Kapitel + Abschnitt + Artikel)"""
    logger.info(f"Gruppiere {len(documents)} Dokumente nach Artikeln...")

    grouped = {}

    for doc in documents:
        metadata = doc.get('metadata', {})

        # Eindeutiger Schlüssel
        key = f"kap_{metadata.get('kapitel_nr', 0)}_abs_{metadata.get('abschnitt_nr', 0)}_art_{metadata.get('artikel_nr', 0)}"

        if key not in grouped:
            grouped[key] = {
                'id': key,
                'texts': [],
                'metadata': {
                    'kapitel_nr': metadata.get('kapitel_nr', 0),
                    'kapitel_name': metadata.get('kapitel_name', ''),
                    'abschnitt_nr': metadata.get('abschnitt_nr', 0),
                    'abschnitt_name': metadata.get('abschnitt_name', ''),
                    'artikel_nr': metadata.get('artikel_nr', 0),
                    'artikel_name': metadata.get('artikel_name', ''),
                    'grouped': True
                }
            }

        # Text hinzufügen
        if doc.get('text', '').strip():
            grouped[key]['texts'].append(doc['text'].strip())

    # Ergebnis erstellen
    result = []
    for group in grouped.values():
        if group['texts']:  # Nur wenn Texte vorhanden
            result.append({
                'id': group['id'],
                'text': '\n\n'.join(group['texts']),
                'metadata': group['metadata']
            })

    logger.info(f"Gruppierung abgeschlossen: {len(documents)} → {len(result)} Dokumente")
    return result


def apply_grouping_if_enabled(documents: List[Dict[str, Any]],
                              config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Gruppiere Dokumente falls in Config aktiviert"""
    grouping_config = config.get('grouping', {})

    if not grouping_config.get('enabled', False):
        logger.info("Dokumenten-Gruppierung ist deaktiviert")
        return documents

    return group_documents_by_article(documents)