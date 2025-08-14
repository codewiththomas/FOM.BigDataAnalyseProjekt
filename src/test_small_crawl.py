#!/usr/bin/env python3
"""
Test script for small crawl functionality
"""

import logging
import json
from dsgvo_crawler import DSGVOCrawler

def test_small_crawl():
    """Test crawling with just a few articles"""
    logging.basicConfig(level=logging.WARNING)

    crawler = DSGVOCrawler()

    # Get overview page
    overview_soup = crawler.http_client.get_page(crawler.base_url)
    if not overview_soup:
        print("Could not load overview page")
        return

    # Get first 3 articles
    artikel_links = crawler.overview_parser.parse_overview_page(overview_soup)
    if not artikel_links:
        print("No articles found")
        return

    print(f"Found {len(artikel_links)} articles, testing first 3...")

    all_entries = []

    for i, (artikel_url, context) in enumerate(artikel_links[:3], 1):
        print(f"\nProcessing article {i}: {artikel_url}")

        soup = crawler.http_client.get_page(artikel_url)
        if soup:
            entries = crawler.article_parser.parse_article_page(soup, artikel_url, context)
            all_entries.extend(entries)
            print(f"  Extracted {len(entries)} entries")
        else:
            print(f"  Could not load article")

    print(f"\nTotal entries extracted: {len(all_entries)}")

    # Show sample entries
    print("\nSample entries:")
    for i, entry in enumerate(all_entries[:5], 1):
        print(f"{i}. {entry._generate_id()}")
        print(f"   Text: {entry.text[:100]}...")
        print()

    # Check for any problematic sentence numbers
    problematic_entries = [e for e in all_entries if e.satz_nr > 100]
    if problematic_entries:
        print(f"WARNING: Found {len(problematic_entries)} entries with high sentence numbers")
    else:
        print("OK: No problematic sentence numbers found")

    # Test ID generation
    print("\nTesting ID generation:")
    for entry in all_entries[:3]:
        print(f"  {entry._generate_id()}")

if __name__ == "__main__":
    test_small_crawl()
