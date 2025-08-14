from dataclasses import dataclass
from typing import Dict

@dataclass
class DsgvoElement:
    id: str
    kapitel_nr: int
    kapitel_name: str
    abschnitt_nr: int
    abschnitt_name: str
    artikel_nr: int
    artikel_name: str
    absatz_nr: int
    unterabsatz_nr: int
    satz_nr: int
    text: str

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'kapitel_nr': self.kapitel_nr,
            'kapitel_name': self.kapitel_name,
            'abschnitt_nr': self.abschnitt_nr,
            'abschnitt_name': self.abschnitt_name,
            'artikel_nr': self.artikel_nr,
            'artikel_name': self.artikel_name,
            'absatz_nr': self.absatz_nr,
            'unterabsatz_nr': self.unterabsatz_nr,
            'satz_nr': self.satz_nr,
            'text': self.text
        }


from bs4 import BeautifulSoup
from datetime import datetime
from typing import Optional, List, Tuple

import json
import logging
import re
import requests
import time


class DsgvoCrawler:

    def __init__(self, logger: logging.Logger, base_url: str = "https://dsgvo-gesetz.de"):
        self.logger = logger
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        })
        self.aktuelles_kapitel = { "nr": 0, "name": "" }
        self.aktueller_abschnitt = { "nr": 0, "name": "" }
        self.logger.info(f"DsgvoCrawler initialisiert mit Base-URL: {self.base_url}")


    def _get_page(self, url: str, retry_count: int = 3) -> Optional[BeautifulSoup]:
        "Hilfsmethode zum laden einer Seite mit Retry-Mechanismus"
        for attempt in range(retry_count):
            try:
                self.logger.info(f"Lade Seite: {url} (Versuch {attempt + 1})")
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'lxml')
                time.sleep(1)
                return soup
            except requests.RequestException as e:
                self.logger.warning(f"Fehler beim Laden von {url}: {e}")
                if attempt == retry_count - 1:
                    self.logger.error(f"Alle Versuche fehlgeschlagen für: {url}")
                    return None
                time.sleep(2 ** attempt)
        return None


    def _parse_overview_page(self) -> List[Tuple[str, dict]]:
        """
        """
        soup = self._get_page(self.base_url)
        if not soup:
            self.logger.error("Übersichtsseite konnte nicht geladen werden")
            return []

        # Inhaltsverzeichnis suchen
        inhaltsverzeichnis = soup.find('div', class_='liste-inhaltsuebersicht dsgvo')
        if not inhaltsverzeichnis:
            logger.error("Inhaltsverzeichnis nicht gefunden")
            return []
        self.logger.info(f"Inhaltsverzeichnis gefunden.")

        artikel_links = []

        # Alle Elemente im Inhaltsverzeichnis durchlaufen
        for element in inhaltsverzeichnis.find_all(['div']):

            class_name = element.get('class', [])

            # Neues Kapitel
            if 'kapitel' in class_name:
                kapitel_nummer_span = element.find('span', class_='nummer')
                kapitel_titel_span = element.find('span', class_='titel')
                if kapitel_nummer_span and kapitel_titel_span:
                    kapitel_text = kapitel_nummer_span.get_text(strip=True)
                    kapitel_match = re.search(r'(\d+)', kapitel_text)
                    self.aktuelles_kapitel['nr'] = int(kapitel_match.group(1)) if kapitel_match else 0
                    self.aktuelles_kapitel['name'] = kapitel_titel_span.get_text(strip=True)
                    self.logger.info(f"Neues Kapitel: {self.aktuelles_kapitel['nr']} {self.aktuelles_kapitel['name']}")
                else:
                    self.logger.error("Fehler beim Extrahieren des Kapitel-Textes")
                    continue

            # Neuer Abschnitt unterhalb des aktuellen Kapitels
            elif 'abschnitt' in class_name:
                abschnitt_nummer_span = element.find('span', class_='nummer')
                abschnitt_titel_span = element.find('span', class_='titel')
                if abschnitt_nummer_span and abschnitt_titel_span:
                    abschnitt_text = abschnitt_nummer_span.get_text(strip=True)
                    abschnitt_match = re.search(r'(\d+)', abschnitt_text)
                    self.aktueller_abschnitt['nr'] = int(abschnitt_match.group(1)) if abschnitt_match else 0
                    self.aktueller_abschnitt['name'] = abschnitt_titel_span.get_text(strip=True)
                    self.logger.info(f"Neuer Abschnitt: {self.aktueller_abschnitt['nr']} {self.aktueller_abschnitt['name']}")
                else:
                    self.logger.error("Fehler beim Extrahieren des Abschnitt-Textes")

            # Neuer Artikel unterhalb eines Kapitels (und ggf. Abschnitts)
            elif 'artikel' in class_name:
                link = element.find('a')
                if link and link.get('href'):
                    artikel_url = link.get('href')
                    artikel_context = {
                        'kapitel_nr': self.aktuelles_kapitel['nr'],
                        'kapitel_name': self.aktuelles_kapitel['name'],
                        'abschnitt_nr': self.aktueller_abschnitt['nr'],
                        'abschnitt_name': self.aktueller_abschnitt['name']
                    }
                    artikel_links.append((artikel_url, artikel_context))
                    self.logger.info(f"Neuer Artikel: {artikel_url}")
                else:
                    self.logger.error("Fehler beim Extrahieren des Artikel-Links")

        self.logger.info(f"Gefundene Artikel: {len(artikel_links)}")
        return artikel_links



    def _parse_article_page(self, artikel_url: str, context: dict) -> List[DsgvoElement]:
        """
        Parst eine einzelne Artikelseite und extrahiert alle relevanten Elemente
        """
        soup = self._get_page(artikel_url)
        if not soup:
            self.logger.error(f"Artikelseite {artikel_url} konnte nicht geladen werden")
            return []

        eintraege = []

        # Extrahiere Artikel-Info aus dem Header
        header = soup.find('header', class_='entry-header')
        if not header:
            self.logger.error(f"Header nicht gefunden in: {artikel_url}")
            return []

        # Artikelname und -nummer aus dem Header
        artikel_nummer_span = header.find('span', class_='dsgvo-number')
        artikel_titel_span = header.find('span', class_='dsgvo-title')

        if not artikel_nummer_span or not artikel_titel_span:
            self.logger.error(f"Artikel-Info nicht gefunden in: {url}")
            return []


        return []


    def _crawl_all_articles(self) -> List[DsgvoElement]:
        """
        """
        self.logger.info("Starte Crawling aller Artikel...")

        artikel_links = self._parse_overview_page()
        if not artikel_links:
            self.logger.error("Keine Artikel gefunden")
            return []

        all_entries = []

        for i, (artikel_url, context) in enumerate(artikel_links, 1):
            self.logger.info(f"Verarbeite Artikel {i}/{len(artikel_links)}: {artikel_url}")

            try:
                entries = self._parse_article_page(artikel_url, context)
                all_entries.extend(entries)
                self.logger.info(f"Artikel {i}: {len(entries)} Einträge extrahiert")
            except Exception as e:
                self.logger.error(f"Fehler beim Verarbeiten von {artikel_url}: {e}")
                continue

        self.logger.info(f"Crawling abgeschlossen. Insgesamt {len(all_entries)} Einträge extrahiert.")
        return all_entries


    def run(self, output_path: str = None):
        """
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
            output_path = f"../data/output/dsgvo_crawled_{timestamp}.jsonl"

        try:
            entries = self._crawl_all_articles()
            return output_path

        except Exception as e:
            self.logger.error(f"Fehler beim Crawling: {e}")
            return None




if (__name__ == "__main__"):

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../data/logs/crawler.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    crawler = DsgvoCrawler(logger)
    crawler.run()
