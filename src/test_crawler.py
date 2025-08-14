#!/usr/bin/env python3
"""
Test script for DSGVO Crawler

This script tests the basic functionality of the DSGVO crawler
without running a full crawl.
"""

import logging
from dsgvo_crawler import DSGVOCrawler, setup_logging

def test_crawler_initialization():
    """Test if crawler can be initialized"""
    try:
        crawler = DSGVOCrawler()
        print("✓ Crawler initialization successful")
        return True
    except Exception as e:
        print(f"✗ Crawler initialization failed: {e}")
        return False

def test_overview_page_parsing():
    """Test if overview page can be parsed"""
    try:
        crawler = DSGVOCrawler()
        overview_soup = crawler.http_client.get_page(crawler.base_url)

        if overview_soup:
            artikel_links = crawler.overview_parser.parse_overview_page(overview_soup)
            print(f"✓ Overview page parsing successful: {len(artikel_links)} articles found")
            return True
        else:
            print("✗ Could not load overview page")
            return False
    except Exception as e:
        print(f"✗ Overview page parsing failed: {e}")
        return False

def test_single_article_parsing():
    """Test if a single article can be parsed"""
    try:
        crawler = DSGVOCrawler()

        # Test with article 1
        test_url = "https://dsgvo-gesetz.de/art-1-dsgvo/"
        soup = crawler.http_client.get_page(test_url)

        if soup:
            # Create a simple context for testing
            context = {
                'kapitel_nr': 1,
                'kapitel_name': 'Allgemeine Bestimmungen',
                'abschnitt_nr': 0,
                'abschnitt_name': ''
            }

            entries = crawler.article_parser.parse_article_page(soup, test_url, context)
            print(f"✓ Single article parsing successful: {len(entries)} entries extracted")

            # Show first entry details
            if entries:
                first_entry = entries[0]
                print(f"  First entry ID: {first_entry._generate_id()}")
                print(f"  Text preview: {first_entry.text[:100]}...")

            return True
        else:
            print("✗ Could not load test article")
            return False
    except Exception as e:
        print(f"✗ Single article parsing failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing DSGVO Crawler...\n")

    tests = [
        test_crawler_initialization,
        test_overview_page_parsing,
        test_single_article_parsing
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All tests passed! The crawler is ready to use.")
    else:
        print("✗ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    # Setup basic logging for tests
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
    main()
