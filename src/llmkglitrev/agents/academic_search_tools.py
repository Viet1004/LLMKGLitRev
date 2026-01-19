"""
Academic search tools using Google Scholar and arXiv MCP servers.

Provides focused academic paper search across multiple sources including:
- Google Scholar (covers IEEE, Springer, ACM, Nature, Science, etc.)
- arXiv (preprints in physics, CS, math, stats, etc.)

These tools replace Tavily for academic content to improve search quality
and focus on peer-reviewed literature.
"""

from typing import List, Dict, Optional
from difflib import SequenceMatcher
import re
from langchain_core.tools import tool


# ===== Internal Helper Functions (return List[Dict]) =====

def _search_google_scholar_internal(
    query: str,
    author: Optional[str] = None,
    year_low: Optional[int] = None,
    year_high: Optional[int] = None,
    num_results: int = 10
) -> List[Dict]:
    """Internal Google Scholar search that returns List[Dict]."""
    try:
        from google_scholar_mcp import search_google_scholar as gs_search
    except ImportError:
        return []

    try:
        results = gs_search(
            query=query,
            author=author,
            year_low=year_low,
            year_high=year_high,
            num_results=num_results
        )

        formatted_papers = []
        for paper in results:
            formatted_papers.append({
                "title": paper.get("title", "Untitled"),
                "authors": paper.get("authors", []),
                "year": paper.get("year"),
                "venue": paper.get("venue", ""),
                "abstract": paper.get("abstract", ""),
                "citations": paper.get("citations", 0),
                "url": paper.get("url", ""),
                "bibtex": paper.get("bibtex", ""),
                "source": "google_scholar",
                "relevance_score": _calculate_relevance_score(paper)
            })

        return formatted_papers

    except Exception as e:
        print(f"❌ Google Scholar search error: {e}")
        return []


def _search_arxiv_internal(
    query: str,
    max_results: int = 10,
    date_from: Optional[str] = None,
    categories: Optional[List[str]] = None
) -> List[Dict]:
    """Internal arXiv search that returns List[Dict]."""
    print("-1:::===================")
    try:
        import arxiv

    except ImportError:
        print("-0.5:::===================")
        return []
    print("0:::===================")
    try:
        print("1:::===================")
        search_query = query
        if categories:
            category_filter = " OR ".join([f"cat:{c}" for c in categories])
            search_query = f"{query} AND ({category_filter})"
        print("2:::===================")
        search = arxiv.Search(
            query=search_query,
            max_results=max_results * 2,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = []
        print("3::::===================")
        print(search)
        for paper in search.results():
            
            if date_from:
                from datetime import datetime
                pub_date = paper.published.date()
                min_date = datetime.strptime(date_from, "%Y-%m-%d").date()
                if pub_date < min_date:
                    continue

            results.append({
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "year": paper.published.year,
                "abstract": paper.summary,
                "url": paper.entry_id,
                "pdf_url": paper.pdf_url,
                "arxiv_id": paper.entry_id.split("/")[-1],
                "categories": paper.categories,
                "venue": "arXiv",
                "source": "arxiv",
                "relevance_score": 0.5
            })

            if len(results) >= max_results:
                break

        return results

    except Exception as e:
        print(f"❌ arXiv search error: {e}")
        return []


# ===== Public Tool Functions (decorated with @tool, return str) =====

@tool(parse_docstring=True)
def search_google_scholar(
    query: str,
    author: Optional[str] = None,
    year_low: Optional[int] = None,
    year_high: Optional[int] = None,
    num_results: int = 10
) -> str:
    """
    Search Google Scholar for academic papers.

    Coverage: IEEE, Springer, ACM, Nature, Science, arXiv, PubMed, and all
    major academic publishers indexed by Google Scholar.

    Args:
        query: Search keywords (natural language)
        author: Optional author name filter
        year_low: Minimum publication year
        year_high: Maximum publication year
        num_results: Number of results (default 10, max 20 recommended)

    Returns:
        List of papers with:
        - title: Paper title
        - authors: List of author names
        - year: Publication year
        - venue: Journal/conference name
        - abstract: Full abstract text
        - citations: Citation count
        - url: Link to paper
        - bibtex: BibTeX citation format
        - source: "google_scholar"

    Example:
        >>> papers = search_google_scholar(
        ...     query="attention is all you need",
        ...     author="Vaswani",
        ...     year_low=2017,
        ...     num_results=5
        ... )
    """
    # Check if package is installed
    try:
        import google_scholar_mcp
    except ImportError:
        return "⚠️ Google Scholar MCP not installed. Install with: pip install git+https://github.com/arrogant-R/google-scholar-mcp.git"

    # Call internal function
    formatted_papers = _search_google_scholar_internal(
        query=query,
        author=author,
        year_low=year_low,
        year_high=year_high,
        num_results=num_results
    )

    # Format as string for LLM
    if not formatted_papers:
        return "No papers found for the given query in Google Scholar."

    output = f"Found {len(formatted_papers)} papers from Google Scholar:\n\n"
    for i, paper in enumerate(formatted_papers, 1):
        output += f"--- PAPER {i} ---\n"
        output += format_paper_for_llm(paper)
        output += "\n" + "=" * 80 + "\n\n"

    return output


# ===== arXiv Search =====

@tool(parse_docstring=True)
def search_arxiv(
    query: str,
    max_results: int = 10,
    date_from: Optional[str] = None,
    categories: Optional[List[str]] = None
) -> str:
    """
    Search arXiv for preprint papers.

    Best for: Latest research in physics, CS, math, stats, econ, bio.

    Args:
        query: Search keywords
        max_results: Number of results (default 10)
        date_from: Filter papers after this date (YYYY-MM-DD format)
        categories: arXiv categories e.g. ["cs.AI", "cs.LG", "cs.CV"] (see https://arxiv.org/category_taxonomy)

    Returns:
        List of papers with:
        - title: Paper title
        - authors: List of author names
        - year: Publication year
        - abstract: Full abstract
        - url: arXiv page URL
        - pdf_url: Direct PDF link
        - arxiv_id: arXiv identifier (e.g., "2301.12345")
        - categories: Subject categories
        - source: "arxiv"

    Example:
        >>> papers = search_arxiv(
        ...     query="transformer architecture",
        ...     max_results=5,
        ...     date_from="2023-01-01",
        ...     categories=["cs.AI", "cs.LG"]
        ... )

    Popular Categories:
        - cs.AI: Artificial Intelligence
        - cs.LG: Machine Learning
        - cs.CV: Computer Vision
        - cs.CL: Computation and Language
        - cs.NE: Neural and Evolutionary Computing
        - stat.ML: Machine Learning (Statistics)
        - physics.*: Various physics categories
        - math.*: Various math categories
    """
    # Check if package is installed
    try:
        import arxiv
    except ImportError:
        return "⚠️ arxiv package not installed. Install with: pip install arxiv"

    # Call internal function
    results = _search_arxiv_internal(
        query=query,
        max_results=max_results,
        date_from=date_from,
        categories=categories
    )

    # Format as string for LLM
    if not results:
        return "No papers found for the given query in arXiv."

    output = f"Found {len(results)} papers from arXiv:\n\n"
    for i, paper in enumerate(results, 1):
        output += f"--- PAPER {i} ---\n"
        output += format_paper_for_llm(paper)
        output += "\n" + "=" * 80 + "\n\n"

    return output


# ===== Combined Multi-Source Search =====

@tool(parse_docstring=True)
def search_academic_papers(
    query: str,
    limit_per_source: int = 5,
    year_min: Optional[int] = None,
    author_filter: Optional[str] = None
) -> str:
    """
    Search multiple academic sources (Google Scholar + arXiv) and combine results.

    This is the PRIMARY TOOL for academic paper search. Use this for comprehensive
    academic literature search. It queries both Google Scholar and arXiv, deduplicates,
    and returns formatted results.

    Args:
        query: Search keywords (natural language, e.g., "few-shot learning computer vision")
        limit_per_source: Max results per source (default 5, max recommended 10)
        year_min: Minimum publication year (e.g., 2020 for papers from 2020 onwards)
        author_filter: Filter by author name for Google Scholar (e.g., "Vaswani")

    Returns:
        Formatted string with papers from both sources, sorted by citations and recency.

    Example:
        >>> search_academic_papers(
        ...     query="few-shot learning computer vision",
        ...     limit_per_source=5,
        ...     year_min=2020
        ... )
    """
    all_papers = []

    # Search Google Scholar
    gs_papers = _search_google_scholar_internal(
        query=query,
        author=author_filter,
        year_low=year_min,
        year_high=None,
        num_results=limit_per_source
    )
    all_papers.extend(gs_papers)

    # Search arXiv
    date_from = f"{year_min}-01-01" if year_min else None
    # Note: No category filtering - let arXiv's search handle relevance
    # This ensures the tool works for ALL academic domains, not just AI/ML/CV/NLP
    arxiv_papers = _search_arxiv_internal(
        query=query,
        max_results=limit_per_source,
        date_from=date_from,
        categories=None  # No category restriction
    )
    all_papers.extend(arxiv_papers)

    # Deduplicate based on title similarity
    unique_papers = _deduplicate_papers(all_papers)

    # Sort by relevance (citations + recency)
    unique_papers.sort(
        key=lambda x: (
            x.get("citations", 0),  # Primary: citation count
            x.get("year", 0),        # Secondary: recency
            x.get("relevance_score", 0)  # Tertiary: relevance
        ),
        reverse=True
    )

    # Format as string for LLM
    if not unique_papers:
        return "No academic papers found for the given query."

    output = f"Found {len(unique_papers)} unique academic papers (combined from Google Scholar and arXiv):\n\n"

    # Group by source for clarity
    gs_papers_list = [p for p in unique_papers if p['source'] == 'google_scholar']
    arxiv_papers_list = [p for p in unique_papers if p['source'] == 'arxiv']

    output += f"📊 Summary: {len(gs_papers_list)} from Google Scholar, {len(arxiv_papers_list)} from arXiv\n\n"
    output += "=" * 80 + "\n\n"

    for i, paper in enumerate(unique_papers, 1):
        output += f"--- PAPER {i} [{paper['source'].upper()}] ---\n"
        output += format_paper_for_llm(paper)
        output += "\n" + "=" * 80 + "\n\n"

    return output


# ===== Helper Functions =====

def _deduplicate_papers(papers: List[Dict]) -> List[Dict]:
    """
    Remove duplicate papers based on title similarity.

    Uses 85% similarity threshold to catch near-duplicates.
    """
    unique_papers = []
    seen_titles = []

    for paper in papers:
        title = paper.get("title", "").lower().strip()

        if not title:
            continue

        # Check similarity with existing titles
        is_duplicate = False
        for seen_title in seen_titles:
            similarity = SequenceMatcher(None, title, seen_title).ratio()
            if similarity > 0.85:  # 85% similarity = duplicate
                is_duplicate = True
                break

        if not is_duplicate:
            unique_papers.append(paper)
            seen_titles.append(title)

    return unique_papers


def _calculate_relevance_score(paper: Dict) -> float:
    """
    Calculate relevance score for a paper based on multiple factors.

    Factors:
    - Citation count (normalized) - 40% weight
    - Recency (newer = higher) - 30% weight
    - Venue quality (top venues = higher) - 20% weight
    - Abstract length (complete = higher) - 10% weight

    NOTE: The venue quality list is biased toward AI/ML/CV/NLP and general science.
    Papers from other domains (physics, math, economics, etc.) won't get the venue
    bonus, but this only affects 20% of the score. Citation count and recency are
    domain-agnostic and make up 70% of the score.

    Returns score between 0 and 1.
    """
    score = 0.0

    # Citations (max 100 for normalization) - PRIMARY SIGNAL (40%)
    citations = paper.get("citations", 0)
    score += min(citations / 100.0, 1.0) * 0.4

    # Recency (papers from last 5 years get bonus) - SECONDARY SIGNAL (30%)
    year = paper.get("year")
    if year:
        from datetime import datetime
        current_year = datetime.now().year
        years_old = current_year - year
        if years_old <= 5:
            score += (1.0 - years_old / 5.0) * 0.3

    # Venue quality (top venues get bonus) - TERTIARY SIGNAL (20%)
    # NOTE: Limited to common venues in AI/ML/CV/NLP + general science
    # For other domains, the citation count is more reliable anyway
    venue = paper.get("venue", "").lower()
    top_venues = [
        "nature", "science", "cell", "lancet",           # General science + medical
        "neurips", "icml", "iclr", "cvpr", "iccv", "eccv",  # AI/ML/CV conferences
        "acl", "emnlp", "naacl", "coling",                # NLP conferences
        "ieee", "acm", "springer"                         # Major publishers
    ]
    if any(v in venue for v in top_venues):
        score += 0.2

    # Abstract completeness - MINOR SIGNAL (10%)
    abstract = paper.get("abstract", "")
    if len(abstract) > 100:
        score += 0.1

    return min(score, 1.0)


def get_paper_citation_info(paper: Dict) -> str:
    """
    Format paper as citation string.

    Example output:
        "Vaswani et al. (2017). Attention is All You Need. NeurIPS."
    """
    authors = paper.get("authors", [])
    if not authors:
        author_str = "Unknown"
    elif len(authors) == 1:
        author_str = authors[0]
    elif len(authors) == 2:
        author_str = f"{authors[0]} and {authors[1]}"
    else:
        # Get first author's last name if possible
        first_author = authors[0]
        if "," in first_author:
            last_name = first_author.split(",")[0]
        else:
            last_name = first_author.split()[-1]
        author_str = f"{last_name} et al."

    year = paper.get("year", "n.d.")
    title = paper.get("title", "Untitled")
    venue = paper.get("venue", "")

    citation = f"{author_str} ({year}). {title}."
    if venue:
        citation += f" {venue}."

    return citation


def format_paper_for_llm(paper: Dict) -> str:
    """
    Format paper information for LLM consumption.

    Provides structured, readable format with all key information.
    """
    lines = []

    # Title
    lines.append(f"**Title:** {paper.get('title', 'Untitled')}")

    # Authors
    authors = paper.get("authors", [])
    
    author_str = ", ".join(authors)
    
    lines.append(f"**Authors:** {author_str}")

    # Year and Venue
    year = paper.get("year", "Unknown")
    venue = paper.get("venue", "Unknown venue")
    lines.append(f"**Published:** {year} | {venue}")

    # Citations (if available)
    citations = paper.get("citations")
    if citations is not None:
        lines.append(f"**Citations:** {citations}")

    # Abstract
    abstract = paper.get("abstract", "No abstract available")

    lines.append(f"**Abstract:** {abstract}")

    # URL
    url = paper.get("url", "")
    if url:
        lines.append(f"**URL:** {url}")

    # PDF (if available)
    pdf_url = paper.get("pdf_url")
    if pdf_url:
        lines.append(f"**PDF:** {pdf_url}")

    # Source
    source = paper.get("source", "unknown")
    lines.append(f"**Source:** {source}")

    return "\n".join(lines)


# ===== Utility: Academic Domain Detection =====

def detect_academic_domains(query: str) -> List[str]:
    """
    Detect likely arXiv categories from a query string (OPTIONAL HELPER).

    WARNING: This function has LIMITED COVERAGE and only recognizes:
    - AI/ML (cs.AI, cs.LG, stat.ML)
    - Computer Vision (cs.CV)
    - NLP (cs.CL)
    - Medical/Bio (q-bio.GN, q-bio.QM)

    For queries outside these domains (physics, math, economics, chemistry, etc.),
    it returns an empty list. This is INTENTIONAL - arXiv search works fine without
    category filtering, and letting arXiv's own search algorithm handle relevance
    is more robust than incomplete keyword matching.

    NOTE: This function is NOT used by default in search_academic_papers().
    It's provided as an optional utility for specific use cases.

    Returns:
        List of arXiv category codes, or empty list if no match
    """
    query_lower = query.lower()

    domains = []

    # AI/ML keywords
    ai_ml_keywords = [
        "machine learning", "deep learning", "neural network",
        "artificial intelligence", "reinforcement learning",
        "supervised learning", "unsupervised learning",
        "transformer", "attention", "bert", "gpt"
    ]
    if any(kw in query_lower for kw in ai_ml_keywords):
        domains.extend(["cs.AI", "cs.LG", "stat.ML"])

    # Computer Vision keywords
    cv_keywords = [
        "computer vision", "image", "visual", "video",
        "object detection", "segmentation", "recognition",
        "cnn", "convolutional"
    ]
    if any(kw in query_lower for kw in cv_keywords):
        domains.append("cs.CV")

    # NLP keywords
    nlp_keywords = [
        "natural language", "nlp", "text", "language model",
        "translation", "sentiment", "parsing", "tokenization"
    ]
    if any(kw in query_lower for kw in nlp_keywords):
        domains.append("cs.CL")

    # Medical/Bio keywords
    bio_keywords = [
        "medical", "clinical", "healthcare", "patient",
        "diagnosis", "treatment", "biological", "genomic"
    ]
    if any(kw in query_lower for kw in bio_keywords):
        domains.extend(["q-bio.GN", "q-bio.QM"])

    # NOTE: Many domains are NOT covered here:
    # - Physics (physics.*): quantum mechanics, astrophysics, etc.
    # - Mathematics (math.*): algebra, topology, number theory, etc.
    # - Economics (econ.*): econometrics, game theory, etc.
    # - Statistics (stat.*): methodology, theory, etc.
    # - Quantitative Finance (q-fin.*): portfolio management, risk, etc.
    # - And many more...
    #
    # For these domains, return empty list and let arXiv search handle it

    return list(set(domains))  # Remove duplicates
