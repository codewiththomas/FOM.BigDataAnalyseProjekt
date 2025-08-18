from typing import List, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


def group_documents_by_article(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Gruppiere DSGVO-Dokumente nach Artikel (Kapitel + Abschnitt + Artikel)
    @param documents: Eine Liste an Dokumenten im Format List[Dict[str, Any]]
                      Beispiel [{'id': '1', 'text': 'Text des Dokuments', 'metadata': {'kapitel_nr': 1, 'kapitel_name': 'Kapitel 1', 'abschnitt_nr': 1, 'abschnitt_name': 'Abschnitt 1', 'artikel_nr': 1, 'artikel_name': 'Artikel 1'}}]
    @return: this is a description of what is returned
    """
    logger.info(f"Gruppiere {len(documents)} Dokumente nach Artikeln...")

    grouped = {}

    letzter_artikel = 0      # Artikel x
    letzter_absatz = 0       # Abs. 1
    letzter_unterabsatz = 0  # (a)
    letzter_satz = 1         # [Satz 1]

    for doc in documents:

        # print(doc)
        # print("Letzter Artikel: " + str(letzter_artikel))
        # print("Letzter Absatz: " + str(letzter_absatz))
        # print("Letzter Unterabsatz: " + str(letzter_unterabsatz))
        # print("Letzter Satz: " + str(letzter_satz))
        # print("--------------------------------")

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
        text = doc.get('text', '').strip()

        # nur text bis zum ersten \n\n
        text = text.split('\n\n')[0]  # Workaround für Fehler, dass im ersten Absatz der gesamte Artikel steht

        if text:

            artikel_prefix_text = ''
            if (metadata.get('artikel_nr', 0) > letzter_artikel):
                artikel_prefix_text = "Art. " + str(metadata.get('artikel_nr', 0)) + " " +str(metadata.get('artikel_name')) + ":\n\n"
                letzter_unterabsatz = 0
                letzter_absatz = 0

            if (metadata.get('unterabsatz_nr', 0) > letzter_unterabsatz):
                unterabsatz_buchstabe = chr(ord('a') + metadata.get('unterabsatz_nr', 0) - 1)
                text = "(" + unterabsatz_buchstabe + ") " + text
            elif (metadata.get('absatz_nr', 0) > letzter_absatz):
                text = "Abs. " + str(metadata.get('absatz_nr', 0)) + " " + text

            # Satznummer wird nicht geschrieben, da bei Zusammenfassung irrelevant

            if artikel_prefix_text:
                text = artikel_prefix_text + text

            letzter_unterabsatz = metadata.get('unterabsatz_nr', 0)
            letzter_absatz = metadata.get('absatz_nr', 0)
            letzter_artikel = metadata.get('artikel_nr', 0)

            grouped[key]['texts'].append(text)

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

    result_path = 'data/output/dsgvo_grouped.txt'
    logger.info(f"Speichere gruppierten Dokumentenkorpus in {result_path} (nur zum Debuggen erforderlich).")

    with open(result_path, "w") as datei:
        datei.write(json.dumps(result))

    return result


def apply_grouping_if_enabled(documents: List[Dict[str, Any]],
                              config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Gruppiere Dokumente falls in Config aktiviert"""
    grouping_config = config.get('grouping', {})

    if not grouping_config.get('enabled', False):
        logger.info("Dokumenten-Gruppierung ist deaktiviert")
        return documents

    return group_documents_by_article(documents)