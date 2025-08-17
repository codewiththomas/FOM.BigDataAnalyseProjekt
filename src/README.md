# DSGVO Crawler

A clean, well-structured Python crawler for extracting DSGVO (GDPR) articles from https://dsgvo-gesetz.de/ and converting them to structured JSONL format.

## Features

- **Clean Architecture**: Follows SOLID principles with separate classes for different responsibilities
- **DRY Code**: No code duplication, reusable components
- **Structured Output**: Generates JSONL with hierarchical structure (chapters, sections, articles, paragraphs, sentences)
- **Unique IDs**: Each entry gets a unique identifier in the format `dsgvo_art_{artikel_nr}_abs_{absatz_nr}({unterabs_nr})_satz_{satz_nr}`
- **Robust Parsing**: Handles complex article structures with nested paragraphs and subparagraphs
- **Error Handling**: Comprehensive error handling with retry mechanisms
- **Logging**: Detailed logging for debugging and monitoring

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the crawler to extract all DSGVO articles:

```bash
cd src
python dsgvo_crawler.py
```

### Programmatic Usage

```python
from dsgvo_crawler import DSGVOCrawler

# Create crawler instance
crawler = DSGVOCrawler()

# Run full crawl
output_file = crawler.run_full_crawl()

# Or specify custom output path
output_file = crawler.run_full_crawl("custom_output.jsonl")
```

### Testing

Test the crawler functionality without running a full crawl:

```bash
cd src
python test_crawler.py
```

## Output Format

The crawler generates a JSONL file with the following structure:

```json
{
  "id": "dsgvo_art_1_abs_1_satz_1",
  "Kapitel_Nr": 1,
  "Kapitel_Name": "Allgemeine Bestimmungen",
  "Abschnitt_Nr": 0,
  "Abschnitt_Name": "",
  "Artikel_nr": 1,
  "Artikel_Name": "Gegenstand und Ziele",
  "Absatz_nr": 1,
  "Unterabsatz_nr": 0,
  "Satz_nr": 1,
  "Text": "Diese Verordnung enthält Vorschriften zum Schutz natürlicher Personen..."
}
```

### ID Format

Each entry gets a unique ID following this pattern:
- `dsgvo_art_{artikel_nr}_abs_{absatz_nr}({unterabs_nr})_satz_{satz_nr}`

Examples:
- `dsgvo_art_1_abs_1_satz_1` - Article 1, Paragraph 1, Sentence 1
- `dsgvo_art_2_abs_2(1)_satz_1` - Article 2, Paragraph 2, Subparagraph 1, Sentence 1

## Architecture

The crawler follows a clean, modular architecture:

- **`DSGVOCrawler`**: Main orchestrator class
- **`HTTPClient`**: Handles HTTP requests with retry logic
- **`OverviewPageParser`**: Parses the overview page structure
- **`ArticlePageParser`**: Parses individual article pages
- **`ContentParser`**: Handles article content parsing
- **`SentenceParser`**: Extracts sentences from HTML/text
- **`DataExporter`**: Handles data export to JSONL

## Logging

The crawler creates detailed logs in `../data/logs/crawler.log` and also outputs to console.

## Output Files

- **Data**: `../data/output/dsgvo_crawled_{timestamp}.jsonl`
- **Logs**: `../data/logs/crawler.log`

## Error Handling

- Retry mechanism for failed HTTP requests
- Graceful handling of parsing errors
- Comprehensive logging of all operations
- Continues processing even if individual articles fail

## Rate Limiting

The crawler includes a 1-second delay between requests to be respectful to the target website.

## Dependencies

- `requests`: HTTP client
- `beautifulsoup4`: HTML parsing
- `lxml`: XML/HTML parser backend
- Standard library modules: `json`, `logging`, `re`, `time`, `datetime`, `pathlib`, `typing`, `dataclasses`
