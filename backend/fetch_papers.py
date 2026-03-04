"""
fetch_papers.py  Fetch research papers from the arXiv API and parse results.
"""
import requests
import xml.etree.ElementTree as ET
from datetime import datetime

ARXIV_API_URL = "http://export.arxiv.org/api/query"
NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}


def fetch_arxiv(query: str, max_results: int = 12) -> list[dict]:
    """
    Query the arXiv API and return a list of structured paper dicts.

    Each dict contains:
      - id         : arXiv identifier string
      - title      : paper title (cleaned)
      - authors    : comma-separated author names
      - abstract   : full abstract text
      - published  : ISO date string (YYYY-MM-DD)
      - url        : link to the abstract page on arxiv.org
    """
    params = {
        "search_query": f"all:{query}",
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    try:
        response = requests.get(ARXIV_API_URL, params=params, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"arXiv API request failed: {exc}") from exc

    root = ET.fromstring(response.text)
    papers = []

    for entry in root.findall("atom:entry", NAMESPACE):
        # ── title ────────────────────────────────────────────────────────────
        title_el = entry.find("atom:title", NAMESPACE)
        title = " ".join(title_el.text.split()) if title_el is not None else "Untitled"

        # ── authors ──────────────────────────────────────────────────────────
        author_names = []
        for author_el in entry.findall("atom:author", NAMESPACE):
            name_el = author_el.find("atom:name", NAMESPACE)
            if name_el is not None:
                author_names.append(name_el.text.strip())
        authors = ", ".join(author_names) if author_names else "Unknown"

        # ── abstract ─────────────────────────────────────────────────────────
        abstract_el = entry.find("atom:summary", NAMESPACE)
        abstract = (
            " ".join(abstract_el.text.split()) if abstract_el is not None else ""
        )

        # ── published date ────────────────────────────────────────────────────
        published_el = entry.find("atom:published", NAMESPACE)
        published = ""
        if published_el is not None:
            try:
                published = datetime.fromisoformat(
                    published_el.text.replace("Z", "+00:00")
                ).strftime("%Y-%m-%d")
            except ValueError:
                published = published_el.text[:10]

        # ── arXiv ID & URL ────────────────────────────────────────────────────
        id_el = entry.find("atom:id", NAMESPACE)
        url = id_el.text.strip() if id_el is not None else ""
        arxiv_id = url.split("/abs/")[-1] if "/abs/" in url else url

        papers.append(
            {
                "id": arxiv_id,
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "published": published,
                "url": url,
            }
        )

    return papers