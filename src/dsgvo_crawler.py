#!/usr/bin/env python3
"""
DSGVO Website Crawler

A clean, well-structured crawler for https://dsgvo-gesetz.de/ that extracts
DSGVO articles and converts them to structured JSONL format.

Author: Generated from Jupyter notebook
Date: 2025-08-11
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import requests
from bs4 import BeautifulSoup, Tag


@dataclass
class DSGVOEntry:
    """Represents a DSGVO entry with hierarchical structure"""
    kapitel_nr: int
    kapitel_name: str
    abschnitt_nr: int       # 0 if not present
    abschnitt_name: str     # "" if not present
    artikel_nr: int
    artikel_name: str
    absatz_nr: int          # 0 if not explicitly numbered
    unterabsatz_nr: int     # 0 if not explicitly numbered, otherwise a=1, b=2, c=3, ...
    satz_nr: int            # 1 if not explicitly numbered with <sup>#</sup>
    text: str

    def to_dict(self) -> Dict:
        """Convert entry to dictionary for JSONL export"""
        return {
            'id': self._generate_id(),
            'Kapitel_Nr': self.kapitel_nr,
            'Kapitel_Name': self.kapitel_name,
            'Abschnitt_Nr': self.abschnitt_nr,
            'Abschnitt_Name': self.abschnitt_name,
            'Artikel_nr': self.artikel_nr,
            'Artikel_Name': self.artikel_name,
            'Absatz_nr': self.absatz_nr,
            'Unterabsatz_nr': self.unterabsatz_nr,
            'Satz_nr': self.satz_nr,
            'Text': self.text
        }

    def _generate_id(self) -> str:
        """Generate unique ID for the entry"""
        unterabs_part = f"({self.unterabsatz_nr})" if self.unterabsatz_nr > 0 else ""
        return f"dsgvo_art_{self.artikel_nr}_abs_{self.absatz_nr}{unterabs_part}_satz_{self.satz_nr}"


class SentenceParser:
    """Handles sentence parsing from HTML elements and text"""

    @staticmethod
    def parse_sentences_from_element(element) -> List[Tuple[int, str]]:
        """Parse sentences from BeautifulSoup element with <sup> tags"""
        html_str = str(element)

        # Check for <sup> tags
        if '<sup>' in html_str:
            return SentenceParser._parse_sup_tags(html_str)
        else:
            return SentenceParser._parse_numbered_sentences(html_str)

    @staticmethod
    def parse_sentences_from_text(text: str) -> List[Tuple[int, str]]:
        """Parse sentences from text string with <sup> tags"""
        if '<sup>' in text:
            return SentenceParser._parse_sup_tags(text)
        else:
            return SentenceParser._parse_numbered_sentences(text)

    @staticmethod
    def _parse_sup_tags(html_str: str) -> List[Tuple[int, str]]:
        """Parse sentences with <sup> tags"""
        sentences = []
        pattern = r'<sup[^>]*>(\d+)</sup>(.*?)(?=<sup[^>]*>\d+</sup>|$)'
        matches = re.findall(pattern, html_str, re.DOTALL)

        for satz_nr_str, content in matches:
            satz_nr = int(satz_nr_str)
            # Extract only text, remove HTML tags
            satz_text = re.sub(r'<[^>]+>', '', content).strip()
            if satz_text:
                sentences.append((satz_nr, satz_text))

        return sentences

    @staticmethod
    def _parse_numbered_sentences(html_str: str) -> List[Tuple[int, str]]:
        """Parse sentences with number-based sentence numbering"""
        clean_text = re.sub(r'<[^>]+>', '', html_str).strip()
        if not clean_text:
            return []

        # Check if there are <sup> tags for sentence numbering
        if '<sup>' in html_str:
            return SentenceParser._parse_sup_tags(html_str)

        # For now, just return the entire text as sentence 1
        # The complex sentence numbering in DSGVO articles is too difficult to parse reliably
        # without causing false positives from article references, paragraph numbers, etc.
        return [(1, clean_text)]


class ContentParser:
    """Handles parsing of article content and structure"""

    def __init__(self, sentence_parser: SentenceParser):
        self.sentence_parser = sentence_parser

    def parse_article_content(self, content_div: Union[BeautifulSoup, Tag], context: dict, artikel_nr: int, artikel_name: str) -> List[DSGVOEntry]:
        """Parse article content and extract all entries"""
        entries = []

        # Check if it's a simple article (only <p> tags)
        paragraphs = content_div.find_all('p', recursive=False)
        if paragraphs and not content_div.find('ol', recursive=False):
            entries.extend(self._parse_simple_article(paragraphs, context, artikel_nr, artikel_name))
        else:
            # Complex article with numbered paragraphs
            ol_elements = content_div.find_all('ol', recursive=False)
            for ol in ol_elements:
                entries.extend(self._parse_ordered_list(ol, context, artikel_nr, artikel_name))

        return entries

    def _parse_simple_article(self, paragraphs, context: dict, artikel_nr: int, artikel_name: str) -> List[DSGVOEntry]:
        """Parse simple article without numbered paragraphs"""
        entries = []

        for p in paragraphs:
            text = p.get_text(strip=True)
            if text and not self._is_navigation_text(text):
                sentences = self.sentence_parser.parse_sentences_from_element(p)
                for satz_nr, satz_text in sentences:
                    entry = DSGVOEntry(
                        kapitel_nr=context['kapitel_nr'],
                        kapitel_name=context['kapitel_name'],
                        abschnitt_nr=context['abschnitt_nr'],
                        abschnitt_name=context['abschnitt_name'],
                        artikel_nr=artikel_nr,
                        artikel_name=artikel_name,
                        absatz_nr=0,  # No explicit paragraph
                        unterabsatz_nr=0,  # No subparagraph
                        satz_nr=satz_nr,
                        text=satz_text
                    )
                    entries.append(entry)

        return entries

    def _parse_ordered_list(self, ol_element, context: dict, artikel_nr: int, artikel_name: str, parent_absatz: int = 0) -> List[DSGVOEntry]:
        """Parse <ol> list recursively for paragraphs and subparagraphs"""
        entries = []

        for i, li in enumerate(ol_element.find_all('li', recursive=False), 1):
            try:
                absatz_nr = i if parent_absatz == 0 else parent_absatz

                # Check for nested lists (subparagraphs)
                nested_ol = li.find('ol')

                if nested_ol:
                    entries.extend(self._parse_paragraph_with_subparagraphs(
                        li, nested_ol, context, artikel_nr, artikel_name, absatz_nr
                    ))
                else:
                    entries.extend(self._parse_simple_paragraph(
                        li, context, artikel_nr, artikel_name, absatz_nr
                    ))

            except Exception as e:
                logging.error(f"Error parsing paragraph {i} in article {artikel_nr}: {e}")
                continue

        return entries

    def _parse_paragraph_with_subparagraphs(self, li_element, nested_ol, context: dict,
                                          artikel_nr: int, artikel_name: str, absatz_nr: int) -> List[DSGVOEntry]:
        """Parse paragraph that contains subparagraphs"""
        entries = []

        # Extract text before nested list
        li_copy = li_element.__copy__()
        nested_ol_copy = li_copy.find('ol')
        if nested_ol_copy:
            nested_ol_copy.decompose()

        intro_text = li_copy.get_text(strip=True)
        if intro_text:
            # Introduction text of paragraph
            sentences = self.sentence_parser.parse_sentences_from_element(li_element)
            for satz_nr, satz_text in sentences:
                entry = DSGVOEntry(
                    kapitel_nr=context['kapitel_nr'],
                    kapitel_name=context['kapitel_name'],
                    abschnitt_nr=context['abschnitt_nr'],
                    abschnitt_name=context['abschnitt_name'],
                    artikel_nr=artikel_nr,
                    artikel_name=artikel_name,
                    absatz_nr=absatz_nr,
                    unterabsatz_nr=0,  # Introduction text
                    satz_nr=satz_nr,
                    text=satz_text
                )
                entries.append(entry)

        # Parse subparagraphs
        for j, sub_li in enumerate(nested_ol.find_all('li', recursive=False), 1):
            text = sub_li.get_text(strip=True)
            if text:
                sentences = self.sentence_parser.parse_sentences_from_text(text)
                for satz_nr, satz_text in sentences:
                    entry = DSGVOEntry(
                        kapitel_nr=context['kapitel_nr'],
                        kapitel_name=context['kapitel_name'],
                        abschnitt_nr=context['abschnitt_nr'],
                        abschnitt_name=context['abschnitt_name'],
                        artikel_nr=artikel_nr,
                        artikel_name=artikel_name,
                        absatz_nr=absatz_nr,
                        unterabsatz_nr=j,  # a=1, b=2, c=3, ...
                        satz_nr=satz_nr,
                        text=satz_text
                    )
                    entries.append(entry)

        return entries

    def _parse_simple_paragraph(self, li_element, context: dict, artikel_nr: int,
                               artikel_name: str, absatz_nr: int) -> List[DSGVOEntry]:
        """Parse simple paragraph without subparagraphs"""
        entries = []

        # Use parse_sentences_from_element to preserve <sup> tags
        sentences = self.sentence_parser.parse_sentences_from_element(li_element)

        for satz_nr, satz_text in sentences:
            entry = DSGVOEntry(
                kapitel_nr=context['kapitel_nr'],
                kapitel_name=context['kapitel_name'],
                abschnitt_nr=context['abschnitt_nr'],
                abschnitt_name=context['abschnitt_name'],
                artikel_nr=artikel_nr,
                artikel_name=artikel_name,
                absatz_nr=absatz_nr,
                unterabsatz_nr=0,
                satz_nr=satz_nr,
                text=satz_text
            )
            entries.append(entry)

        return entries

    @staticmethod
    def _is_navigation_text(text: str) -> bool:
        """Check if text is navigation or meta text"""
        navigation_keywords = [
            'feedback', 'bewertung', 'drucken', 'teilen',
            'weitere artikel', 'siehe auch', 'navigation'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in navigation_keywords)


class OverviewPageParser:
    """Handles parsing of the overview page to extract article structure"""

    def __init__(self):
        self.current_kapitel = {"nr": 0, "name": ""}
        self.current_abschnitt = {"nr": 0, "name": ""}

    def parse_overview_page(self, soup: BeautifulSoup) -> List[Tuple[str, dict]]:
        """Parse overview page and extract all article URLs"""
        # Find table of contents
        inhaltsverzeichnis = soup.find('div', class_='liste-inhaltsuebersicht dsgvo')
        if not inhaltsverzeichnis:
            logging.error("Table of contents not found")
            return []

        artikel_links = []

        # Process all elements in directory
        for element in inhaltsverzeichnis.find_all(['div']):
            class_name = element.get('class', [])

            if 'kapitel' in class_name:
                self._update_kapitel(element)
            elif 'abschnitt' in class_name:
                self._update_abschnitt(element)
            elif 'artikel' in class_name:
                artikel_url = self._extract_artikel_url(element)
                if artikel_url:
                    context = self._create_context()
                    artikel_links.append((artikel_url, context))

        logging.info(f"Total {len(artikel_links)} articles found")
        return artikel_links

    def _update_kapitel(self, element):
        """Update current chapter information"""
        nummer_span = element.find('span', class_='nummer')
        titel_span = element.find('span', class_='titel')

        if nummer_span and titel_span:
            kapitel_text = nummer_span.get_text(strip=True)
            kapitel_match = re.search(r'(\d+)', kapitel_text)
            self.current_kapitel['nr'] = int(kapitel_match.group(1)) if kapitel_match else 0
            self.current_kapitel['name'] = titel_span.get_text(strip=True)

            # Reset section
            self.current_abschnitt = {"nr": 0, "name": ""}

            logging.info(f"Found: Chapter {self.current_kapitel['nr']} - {self.current_kapitel['name']}")

    def _update_abschnitt(self, element):
        """Update current section information"""
        nummer_span = element.find('span', class_='nummer')
        titel_span = element.find('span', class_='titel')

        if nummer_span and titel_span:
            abschnitt_text = nummer_span.get_text(strip=True)
            abschnitt_match = re.search(r'(\d+)', abschnitt_text)
            self.current_abschnitt['nr'] = int(abschnitt_match.group(1)) if abschnitt_match else 0
            self.current_abschnitt['name'] = titel_span.get_text(strip=True)

            logging.info(f"Found: Section {self.current_abschnitt['nr']} - {self.current_abschnitt['name']}")

    def _extract_artikel_url(self, element) -> Optional[str]:
        """Extract article URL from element"""
        link = element.find('a')
        return link.get('href') if link and link.get('href') else None

    def _create_context(self) -> dict:
        """Create context dictionary for article"""
        return {
            'kapitel_nr': self.current_kapitel['nr'],
            'kapitel_name': self.current_kapitel['name'],
            'abschnitt_nr': self.current_abschnitt['nr'],
            'abschnitt_name': self.current_abschnitt['name']
        }


class ArticlePageParser:
    """Handles parsing of individual article pages"""

    def __init__(self, content_parser: ContentParser):
        self.content_parser = content_parser

    def parse_article_page(self, soup: BeautifulSoup, url: str, context: dict) -> List[DSGVOEntry]:
        """Parse individual article page and extract all entries"""
        # Extract article info from header
        header = soup.find('header', class_='entry-header')
        if not header:
            logging.error(f"Header not found in: {url}")
            return []

        # Article number and name
        dsgvo_number = header.find('span', class_='dsgvo-number')
        dsgvo_title = header.find('span', class_='dsgvo-title')

        if not dsgvo_number or not dsgvo_title:
            logging.error(f"Article info not found in: {url}")
            return []

        artikel_nr = self._extract_artikel_number(dsgvo_number)
        artikel_name = dsgvo_title.get_text(strip=True)

        logging.info(f"Parse Article {artikel_nr}: {artikel_name}")

        # Extract content
        content_div = soup.find('div', class_='entry-content')
        if not content_div:
            logging.error(f"Content not found in: {url}")
            return []

        return self.content_parser.parse_article_content(content_div, context, artikel_nr, artikel_name)

    @staticmethod
    def _extract_artikel_number(dsgvo_number_element) -> int:
        """Extract article number from DSGVO number element"""
        artikel_text = dsgvo_number_element.get_text(strip=True)
        artikel_match = re.search(r'(\d+)', artikel_text)
        return int(artikel_match.group(1)) if artikel_match else 0


class HTTPClient:
    """Handles HTTP requests with retry mechanism and rate limiting"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def get_page(self, url: str, retry_count: int = 3) -> Optional[BeautifulSoup]:
        """Load page with retry mechanism"""
        for attempt in range(retry_count):
            try:
                logging.info(f"Loading page: {url} (Attempt {attempt + 1})")
                response = self.session.get(url, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'lxml')
                time.sleep(1)  # Rate limiting
                return soup

            except requests.RequestException as e:
                logging.warning(f"Error loading {url}: {e}")
                if attempt == retry_count - 1:
                    logging.error(f"All attempts failed for: {url}")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff

        return None


class DataExporter:
    """Handles data export to JSONL format"""

    @staticmethod
    def save_to_jsonl(entries: List[DSGVOEntry], output_path: str):
        """Save entries as JSONL file"""
        # Create output directory if not exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        logging.info(f"Saving {len(entries)} entries to: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                json_line = json.dumps(entry.to_dict(), ensure_ascii=False)
                f.write(json_line + '\n')

        logging.info(f"Successfully saved: {output_path}")


class DSGVOCrawler:
    """Main DSGVO Website Crawler"""

    def __init__(self, base_url: str = "https://dsgvo-gesetz.de"):
        self.base_url = base_url

        # Initialize components
        self.http_client = HTTPClient(base_url)
        self.sentence_parser = SentenceParser()
        self.content_parser = ContentParser(self.sentence_parser)
        self.overview_parser = OverviewPageParser()
        self.article_parser = ArticlePageParser(self.content_parser)
        self.data_exporter = DataExporter()

    def crawl_all_articles(self) -> List[DSGVOEntry]:
        """Main method: Crawl all articles and return structured data"""
        logging.info("Starting DSGVO Crawler...")

        # 1. Parse overview page for article links
        overview_soup = self.http_client.get_page(self.base_url)
        if not overview_soup:
            logging.error("Could not load overview page")
            return []

        artikel_links = self.overview_parser.parse_overview_page(overview_soup)
        if not artikel_links:
            logging.error("No articles found")
            return []

        all_entries = []

        # 2. Parse each article
        for i, (artikel_url, context) in enumerate(artikel_links, 1):
            logging.info(f"Processing article {i}/{len(artikel_links)}: {artikel_url}")

            try:
                soup = self.http_client.get_page(artikel_url)
                if soup:
                    entries = self.article_parser.parse_article_page(soup, artikel_url, context)
                    all_entries.extend(entries)
                    logging.info(f"Article {i}: {len(entries)} entries extracted")
                else:
                    logging.error(f"Could not load article: {artikel_url}")

            except Exception as e:
                logging.error(f"Error processing {artikel_url}: {e}")
                continue

        logging.info(f"Crawling completed. Total {len(all_entries)} entries extracted.")
        return all_entries

    def run_full_crawl(self, output_path: Optional[str] = None) -> Optional[str]:
        """Run complete crawl and save results"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
            output_path = f"../data/output/dsgvo_crawled_{timestamp}.jsonl"

        try:
            # Crawl all articles
            entries = self.crawl_all_articles()

            if entries:
                # Save results
                self.data_exporter.save_to_jsonl(entries, output_path)

                # Statistics
                self._log_statistics(entries)
                return output_path
            else:
                logging.error("No entries found to save")
                return None

        except Exception as e:
            logging.error(f"Error during crawling: {e}")
            return None

    def _log_statistics(self, entries: List[DSGVOEntry]):
        """Log crawling statistics"""
        logging.info("=== CRAWLING STATISTICS ===")
        logging.info(f"Total entries: {len(entries)}")

        # Group by articles
        artikel_count = len(set((e.artikel_nr for e in entries)))
        logging.info(f"Articles processed: {artikel_count}")

        # Group by chapters
        kapitel_count = len(set((e.kapitel_nr for e in entries)))
        logging.info(f"Chapters processed: {kapitel_count}")


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../data/logs/crawler.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main function to run the crawler"""
    setup_logging()

    # Create crawler instance
    crawler = DSGVOCrawler()

    # Run complete crawl
    output_file = crawler.run_full_crawl()

    if output_file:
        print(f"\nCrawling successfully completed!")
        print(f"Output file: {output_file}")
        print(f"Log file: ../data/logs/crawler.log")

        # Preview of first entries
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:3]
            print(f"\nFirst {len(lines)} entries:")
            for i, line in enumerate(lines, 1):
                entry = json.loads(line)
                print(f"{i}. Article {entry['Artikel_nr']}, Paragraph {entry['Absatz_nr']}, Sentence {entry['Satz_nr']}")
                print(f"   ID: {entry['id']}")
                print(f"   Text: {entry['Text'][:100]}...")
    else:
        print("Crawling failed!")


if __name__ == "__main__":
    main()
