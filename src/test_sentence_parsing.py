#!/usr/bin/env python3
"""
Test script for sentence parsing functionality
"""

import logging
from dsgvo_crawler import DSGVOCrawler

def test_sentence_parsing():
    """Test sentence parsing with a small sample"""
    logging.basicConfig(level=logging.WARNING)

    crawler = DSGVOCrawler()

    # Test with article 2 (which had the problematic sentence extraction)
    test_url = "https://dsgvo-gesetz.de/art-2-dsgvo/"
    soup = crawler.http_client.get_page(test_url)

    if soup:
        context = {
            'kapitel_nr': 1,
            'kapitel_name': 'Allgemeine Bestimmungen',
            'abschnitt_nr': 0,
            'abschnitt_name': ''
        }

        entries = crawler.article_parser.parse_article_page(soup, test_url, context)

        print(f"Article 2: {len(entries)} entries extracted")
        print("\nDetailed entry analysis:")

        for i, entry in enumerate(entries, 1):
            print(f"{i}. ID: {entry._generate_id()}")
            print(f"   Text: {entry.text}")
            print()

        # Check for problematic sentence numbers
        problematic_entries = [e for e in entries if e.satz_nr > 100]
        if problematic_entries:
            print(f"WARNING: Found {len(problematic_entries)} entries with high sentence numbers:")
            for entry in problematic_entries[:5]:
                print(f"   {entry._generate_id()}: Satz {entry.satz_nr}")
        else:
            print("OK: No problematic sentence numbers found")

    else:
        print("Could not load test article")

if __name__ == "__main__":
    test_sentence_parsing()
