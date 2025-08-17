#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EU AI Act Crawler for EUR-Lex
Crawls the EU AI Act document from EUR-Lex and outputs structured data in JSONL format
"""

import requests
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple
import re

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../data/logs/eu_ai_act_crawler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EUAIActEntry:
    """Data class for EU AI Act entries"""

    def __init__(self,
                 title: str,
                 article_number: str,
                 article_title: str,
                 section: str,
                 subsection: str,
                 paragraph_number: int,
                 subparagraph_number: int,
                 text: str,
                 url: str,
                 timestamp: str):
        self.title = title
        self.article_number = article_number
        self.article_title = article_title
        self.section = section
        self.subsection = subsection
        self.paragraph_number = paragraph_number
        self.subparagraph_number = subparagraph_number
        self.text = text
        self.url = url
        self.timestamp = timestamp

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'title': self.title,
            'article_number': self.article_number,
            'article_title': self.article_title,
            'section': self.section,
            'subsection': self.subsection,
            'paragraph_number': self.paragraph_number,
            'subparagraph_number': self.subparagraph_number,
            'text': self.text,
            'url': self.url,
            'timestamp': self.timestamp
        }


class EUAIActCrawler:
    """EU AI Act Crawler for EUR-Lex website"""

    def __init__(self, base_url: str = "https://eur-lex.europa.eu/legal-content/DE/TXT/HTML/?uri=OJ:L_202401689"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Structure tracking
        self.current_section = ""
        self.current_subsection = ""
        self.current_article = ""
        self.current_article_title = ""

    def get_page(self, url: str, retry_count: int = 3) -> Optional[BeautifulSoup]:
        """Load a page with retry mechanism"""
        for attempt in range(retry_count):
            try:
                logger.info(f"Loading page: {url} (Attempt {attempt + 1})")
                response = self.session.get(url, timeout=15)
                response.raise_for_status()

                # Try to parse as XML first, then fall back to HTML
                try:
                    soup = BeautifulSoup(response.content, 'xml')
                    logger.info("Parsing as XML document")
                except:
                    soup = BeautifulSoup(response.content, 'lxml')
                    logger.info("Parsing as HTML document")

                time.sleep(2)  # Rate limiting for EUR-Lex
                return soup

            except requests.RequestException as e:
                logger.warning(f"Error loading {url}: {e}")
                if attempt == retry_count - 1:
                    logger.error(f"All attempts failed for: {url}")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff

        return None

    def parse_main_page(self) -> List[EUAIActEntry]:
        """Parse the main EU AI Act page and extract structure"""
        soup = self.get_page(self.base_url)
        if not soup:
            logger.error("Could not load main page")
            return []

        logger.info("Parsing main EU AI Act page...")

        # Find the main content area
        main_content = soup.find('div', class_='content') or soup.find('div', id='content') or soup.find('main')
        if not main_content:
            logger.warning("Main content area not found, trying body")
            main_content = soup.find('body')

        if not main_content:
            logger.error("No content area found")
            return []

        # Extract the document title
        title_element = soup.find('h1') or soup.find('title')
        document_title = title_element.get_text(strip=True) if title_element else "EU AI Act"
        logger.info(f"Document title: {document_title}")

        # Parse the content structure
        entries = self._parse_content_structure(main_content, document_title)

        logger.info(f"Extracted {len(entries)} entries from main page")
        return entries

    def _parse_content_structure(self, content_element, document_title: str) -> List[EUAIActEntry]:
        """Parse the content structure recursively"""
        entries = []

        # Look for different types of content elements
        for element in content_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'article'], recursive=False):
            entries.extend(self._process_element(element, document_title))

        return entries

    def _process_element(self, element, document_title: str) -> List[EUAIActEntry]:
        """Process individual elements and extract structured information"""
        entries = []

        tag_name = element.name.lower()

        if tag_name.startswith('h'):
            # Header element - could be section, subsection, or article
            header_text = element.get_text(strip=True)

            if self._is_article_header(header_text):
                # This is an article header
                article_info = self._extract_article_info(header_text)
                if article_info:
                    self.current_article = article_info['number']
                    self.current_article_title = article_info['title']
                    logger.info(f"Found Article {self.current_article}: {self.current_article_title}")

            elif self._is_section_header(header_text):
                # This is a section header
                self.current_section = header_text
                self.current_subsection = ""
                logger.info(f"Found Section: {self.current_section}")

            elif self._is_subsection_header(header_text):
                # This is a subsection header
                self.current_subsection = header_text
                logger.info(f"Found Subsection: {self.current_subsection}")

        elif tag_name == 'p':
            # Paragraph element - extract text content
            text = element.get_text(strip=True)
            if text and len(text) > 10:  # Filter out very short text
                # Check if this text contains structural information
                self._update_structure_from_text(text)

                entry = self._create_entry_from_text(
                    document_title, text, element
                )
                if entry:
                    entries.append(entry)

        elif tag_name in ['div', 'article']:
            # Container element - look for nested content
            nested_entries = self._parse_content_structure(element, document_title)
            entries.extend(nested_entries)

        return entries

    def _is_article_header(self, text: str) -> bool:
        """Check if header text represents an article"""
        article_patterns = [
            r'^Artikel\s+\d+',
            r'^Article\s+\d+',
            r'^Art\.\s*\d+',
            r'^Art\s+\d+',
            r'^Artikel\s+\d+[:\s]+',
            r'^Article\s+\d+[:\s]+'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in article_patterns)

    def _is_section_header(self, text: str) -> bool:
        """Check if header text represents a section"""
        section_patterns = [
            r'^Kapitel\s+\d+',
            r'^Chapter\s+\d+',
            r'^Abschnitt\s+\d+',
            r'^Section\s+\d+',
            r'^KAPITEL\s+\d+',
            r'^CHAPTER\s+\d+',
            r'^ABSCHNITT\s+\d+',
            r'^SECTION\s+\d+'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in section_patterns)

    def _is_subsection_header(self, text: str) -> bool:
        """Check if header text represents a subsection"""
        subsection_patterns = [
            r'^Unterabschnitt\s+\d+',
            r'^Subsection\s+\d+',
            r'^Teil\s+\d+',
            r'^Part\s+\d+'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in subsection_patterns)

    def _extract_article_info(self, text: str) -> Optional[Dict]:
        """Extract article number and title from header text"""
        # Try different patterns for article extraction
        patterns = [
            r'^Artikel\s+(\d+)[:\s]+(.+)',
            r'^Article\s+(\d+)[:\s]+(.+)',
            r'^Art\.\s*(\d+)[:\s]+(.+)',
            r'^Art\s+(\d+)[:\s]+(.+)',
            r'^Artikel\s+(\d+)\s*$',
            r'^Article\s+(\d+)\s*$',
            r'^Art\.\s*(\d+)\s*$',
            r'^Art\s+(\d+)\s*$'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                article_number = match.group(1)
                article_title = match.group(2).strip() if len(match.groups()) > 1 else ""
                return {
                    'number': article_number,
                    'title': article_title
                }

        return None

    def _create_entry_from_text(self, document_title: str, text: str, element) -> Optional[EUAIActEntry]:
        """Create an entry from text content"""
        # Skip navigation or meta text
        if self._is_navigation_text(text):
            return None

        # Extract paragraph and subparagraph numbers if present
        paragraph_num, subparagraph_num = self._extract_paragraph_numbers(element)

        # Create entry
        entry = EUAIActEntry(
            title=document_title,
            article_number=self.current_article or "Unknown",
            article_title=self.current_article_title or "Unknown",
            section=self.current_section or "Unknown",
            subsection=self.current_subsection or "Unknown",
            paragraph_number=paragraph_num,
            subparagraph_number=subparagraph_num,
            text=text,
            url=self.base_url,
            timestamp=datetime.now().isoformat()
        )

        return entry

    def _extract_paragraph_numbers(self, element) -> Tuple[int, int]:
        """Extract paragraph and subparagraph numbers from element"""
        paragraph_num = 0
        subparagraph_num = 0

        # Look for paragraph numbering in the element or its parent
        text = element.get_text()

        # Check for paragraph patterns like (1), (2), (a), (b), etc.
        paragraph_match = re.search(r'\((\d+)\)', text)
        if paragraph_match:
            paragraph_num = int(paragraph_match.group(1))

        # Check for subparagraph patterns like (a), (b), (c), etc.
        subparagraph_match = re.search(r'\(([a-z])\)', text, re.IGNORECASE)
        if subparagraph_match:
            subparagraph_char = subparagraph_match.group(1).lower()
            subparagraph_num = ord(subparagraph_char) - ord('a') + 1

        return paragraph_num, subparagraph_num

    def _update_structure_from_text(self, text: str):
        """Update current structure based on text content"""
        # Check for article references
        article_match = re.search(r'Artikel\s+(\d+)', text, re.IGNORECASE)
        if article_match:
            article_number = article_match.group(1)
            # Try to extract article title if present
            title_match = re.search(r'Artikel\s+\d+[:\s]+(.+)', text, re.IGNORECASE)
            article_title = title_match.group(1).strip() if title_match else ""

            if self.current_article != article_number:
                self.current_article = article_number
                self.current_article_title = article_title
                logger.info(f"Found Article {self.current_article}: {self.current_article_title}")

        # Check for section references
        section_match = re.search(r'KAPITEL\s+(\d+)', text, re.IGNORECASE)
        if section_match:
            section_number = section_match.group(1)
            section_text = f"KAPITEL {section_number}"
            if self.current_section != section_text:
                self.current_section = section_text
                self.current_subsection = ""
                logger.info(f"Found Section: {self.current_section}")

        # Also check for section titles that follow KAPITEL
        if 'KAPITEL' in text and not re.search(r'KAPITEL\s+\d+', text):
            # This might be a section title without a number
            if self.current_section != text:
                self.current_section = text
                self.current_subsection = ""
                logger.info(f"Found Section Title: {self.current_section}")

        # Check for subsection references
        subsection_match = re.search(r'ABSCHNITT\s+(\d+)', text, re.IGNORECASE)
        if subsection_match:
            subsection_number = subsection_match.group(1)
            subsection_text = f"ABSCHNITT {subsection_number}"
            if self.current_subsection != subsection_text:
                self.current_subsection = subsection_text
                logger.info(f"Found Subsection: {self.current_subsection}")

    def _is_navigation_text(self, text: str) -> bool:
        """Check if text is navigation or meta content"""
        navigation_keywords = [
            'navigation', 'menu', 'breadcrumb', 'footer', 'header',
            'copyright', 'legal notice', 'privacy policy', 'terms of use',
            'back to top', 'previous', 'next', 'home'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in navigation_keywords)

    def crawl_document(self) -> List[EUAIActEntry]:
        """Main method: Crawl the EU AI Act document"""
        logger.info("Starting EU AI Act Crawler...")

        try:
            # Parse main page
            entries = self.parse_main_page()

            if entries:
                logger.info(f"Crawling completed. Total entries extracted: {len(entries)}")
                return entries
            else:
                logger.error("No entries found")
                return []

        except Exception as e:
            logger.error(f"Error during crawling: {e}")
            return []

    def save_to_jsonl(self, entries: List[EUAIActEntry], output_path: str):
        """Save entries to JSONL file"""
        # Create output directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {len(entries)} entries to: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                json_line = json.dumps(entry.to_dict(), ensure_ascii=False)
                f.write(json_line + '\n')

        logger.info(f"Successfully saved: {output_path}")

    def run_full_crawl(self, output_path: Optional[str] = None):
        """Run a complete crawl and save results"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
            output_path = f"data/output/eu_ai_act_crawled_{timestamp}.jsonl"

        try:
            # Crawl the document
            entries = self.crawl_document()

            if entries:
                # Save results
                self.save_to_jsonl(entries, output_path)

                # Statistics
                logger.info("=== CRAWLING STATISTICS ===")
                logger.info(f"Total entries: {len(entries)}")

                # Group by articles
                article_count = len(set(e.article_number for e in entries))
                logger.info(f"Articles processed: {article_count}")

                # Group by sections
                section_count = len(set(e.section for e in entries if e.section != "Unknown"))
                logger.info(f"Sections processed: {section_count}")

                return output_path
            else:
                logger.error("No entries to save")
                return None

        except Exception as e:
            logger.error(f"Error during crawling: {e}")
            return None


def main():
    """Main function to run the crawler"""
    # Create crawler instance
    crawler = EUAIActCrawler()

    # Run full crawl
    output_file = crawler.run_full_crawl()

    if output_file:
        print(f"\nCrawling successfully completed!")
        print(f"Output file: {output_file}")
        print(f"Log file: ../data/logs/eu_ai_act_crawler.log")

        # Preview of first entries
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:3]
            print(f"\nFirst {len(lines)} entries:")
            for i, line in enumerate(lines, 1):
                entry = json.loads(line)
                print(f"{i}. Article {entry['article_number']}, Section: {entry['section']}")
                print(f"   Text: {entry['text'][:100]}...")
    else:
        print("Crawling failed!")


if __name__ == "__main__":
    main()