"""Search and web-scraping tools for the Research Agent.

Supported backends:
  - Google Custom Search API  (requires GOOGLE_SEARCH_API_KEY + GOOGLE_SEARCH_ENGINE_ID)
  - Web scraping              (requests + BeautifulSoup)
  - PubMed E-utilities        (no API key required)
"""

import json
import logging
import os

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Google Custom Search
# ---------------------------------------------------------------------------


@tool
def google_search(query: str, max_results: int = 6) -> str:
    """
    Search the web using the Google Custom Search JSON API.

    Requires two environment variables:
      GOOGLE_SEARCH_API_KEY   — your Google Cloud API key
      GOOGLE_SEARCH_ENGINE_ID — your Programmable Search Engine CX identifier

    Args:
        query: The search query string.
        max_results: Number of results to return (1-10, default 6).

    Returns:
        Formatted string with result titles, URLs, and snippets.
    """
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    cx = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key or not cx:
        return (
            "Google Search is not configured. "
            "Set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID in your .env file."
        )

    try:
        endpoint = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cx,
            "q": query,
            "num": min(max_results, 10),
        }
        response = requests.get(endpoint, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        items = data.get("items", [])
        if not items:
            return "No results found for this query."

        formatted = []
        for i, item in enumerate(items, 1):
            formatted.append(
                f"[{i}] {item.get('title', 'No title')}\n"
                f"    URL: {item.get('link', '')}\n"
                f"    {item.get('snippet', '').strip()}\n"
            )
        return "\n".join(formatted)

    except requests.exceptions.HTTPError as exc:
        logger.warning("Google Search HTTP error: %s", exc)
        return f"Google Search API error ({exc.response.status_code}): {exc}"
    except Exception as exc:
        logger.warning("Google Search failed: %s", exc)
        return f"Google Search error: {exc}"


# ---------------------------------------------------------------------------
# Web Scraping
# ---------------------------------------------------------------------------


@tool
def web_scrape(url: str, max_chars: int = 3000) -> str:
    """
    Scrape and extract the main text content from a webpage.

    Removes navigation, scripts, styles, and boilerplate elements to return
    only the readable body content.

    Args:
        url: The URL of the webpage to scrape.
        max_chars: Maximum characters to return (default 3000).

    Returns:
        Extracted text content from the page, truncated to max_chars.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "lxml")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # Prefer article/main content blocks
        content_block = (
            soup.find("article")
            or soup.find("main")
            or soup.find(id="content")
            or soup.find(class_="content")
            or soup.body
        )

        text = content_block.get_text(separator="\n", strip=True) if content_block else ""
        lines = [line for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        if len(clean_text) > max_chars:
            return clean_text[:max_chars] + "\n... [truncated]"
        return clean_text or "No readable content found at the URL."

    except requests.exceptions.Timeout:
        return f"Timeout while fetching {url}"
    except requests.exceptions.HTTPError as exc:
        return f"HTTP error {exc.response.status_code} for {url}"
    except Exception as exc:
        logger.warning("Web scraping failed for %s: %s", url, exc)
        return f"Scraping error: {exc}"


# ---------------------------------------------------------------------------
# PubMed / Medical Database Query
# ---------------------------------------------------------------------------


@tool
def query_medical_database(query: str, database: str = "pubmed") -> str:
    """
    Query a medical / academic database for research article summaries.

    Currently supports PubMed via the NCBI E-utilities API (no key required).

    Args:
        query: Search terms (clinical terms, MeSH headings, author names, etc.).
        database: Database to query. Only 'pubmed' is currently supported.

    Returns:
        JSON string with article titles, authors, journal, publication date, and URL.
    """
    if database.lower() != "pubmed":
        return f"Database '{database}' is not yet supported. Use 'pubmed'."

    try:
        # Step 1: search for IDs
        search_resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": 5, "retmode": "json"},
            timeout=15,
        )
        search_resp.raise_for_status()
        ids = search_resp.json().get("esearchresult", {}).get("idlist", [])

        if not ids:
            return "No articles found in PubMed for this query."

        # Step 2: fetch summaries
        summary_resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=15,
        )
        summary_resp.raise_for_status()
        result_data = summary_resp.json().get("result", {})
        uids = result_data.get("uids", [])

        articles = []
        for uid in uids:
            art = result_data.get(uid, {})
            articles.append(
                {
                    "pmid": uid,
                    "title": art.get("title", ""),
                    "authors": [a.get("name", "") for a in art.get("authors", [])[:3]],
                    "journal": art.get("fulljournalname", ""),
                    "pub_date": art.get("pubdate", ""),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                }
            )

        return json.dumps({"database": "pubmed", "articles": articles}, indent=2)

    except Exception as exc:
        logger.warning("PubMed query failed: %s", exc)
        return f"Database query error: {exc}"


# ---------------------------------------------------------------------------
# Public accessor
# ---------------------------------------------------------------------------


def get_research_tools() -> list:
    """Return the complete list of tools available to the Research Agent."""
    return [google_search, web_scrape, query_medical_database]
