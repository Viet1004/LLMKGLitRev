"""
Paper Reading and Reference Following Module

This module provides functionality for deep research:
1. Reading PDF papers (extracting full text)
2. Extracting references from papers
3. Following reference chains using DFS (Depth-First Search)

Used by character agents during deep research phase.
"""

from typing import List, Dict, Optional, Set
import asyncio
import re
from collections import deque


class PaperReader:
    """
    Reads PDF papers and extracts full text content.

    NOTE: This is a placeholder implementation. In production, you would:
    1. Use pypdf or pdfplumber for PDF parsing
    2. Handle different PDF formats and encodings
    3. Extract structured sections (abstract, intro, methods, results, discussion)
    4. Handle OCR for scanned papers
    """

    def __init__(self):
        self.cache = {}  # Simple cache to avoid re-reading same PDF

    async def read_pdf(self, pdf_url: str) -> Optional[Dict]:
        """
        Read a PDF and extract full text content.

        Args:
            pdf_url: URL to the PDF file

        Returns:
            Dictionary with extracted content:
            {
                "url": str,
                "full_text": str,
                "sections": Dict[str, str],  # section_name -> content
                "references": List[Dict],     # Extracted references
                "success": bool,
                "error": Optional[str]
            }
        """
        # Check cache
        if pdf_url in self.cache:
            return self.cache[pdf_url]

        # Placeholder implementation
        # In production, you would:
        # 1. Download PDF from URL
        # 2. Parse with pypdf/pdfplumber
        # 3. Extract structured content
        # 4. Parse references

        print(f"   📖 Would read PDF: {pdf_url} (placeholder)")

        result = {
            "url": pdf_url,
            "full_text": f"[PDF content from {pdf_url} would be extracted here]",
            "sections": {
                "abstract": "[Abstract would be extracted]",
                "introduction": "[Introduction would be extracted]",
                "methods": "[Methods would be extracted]",
                "results": "[Results would be extracted]",
                "discussion": "[Discussion would be extracted]",
                "references": "[References section would be extracted]"
            },
            "references": [],  # Would be populated by extract_references()
            "success": True,
            "error": None
        }

        # Cache result
        self.cache[pdf_url] = result

        return result

    def extract_references(self, pdf_content: Dict) -> List[Dict]:
        """
        Extract references from PDF content.

        Args:
            pdf_content: Dictionary from read_pdf()

        Returns:
            List of reference dictionaries:
            {
                "title": str,
                "authors": List[str],
                "year": int,
                "venue": str,
                "doi": Optional[str],
                "arxiv_id": Optional[str]
            }
        """
        # Placeholder implementation
        # In production, you would:
        # 1. Parse references section
        # 2. Use regex or ML model to extract structured info
        # 3. Resolve DOIs/arXiv IDs to full paper metadata

        references_text = pdf_content.get("sections", {}).get("references", "")

        # Placeholder: return empty list
        # Real implementation would parse references_text
        return []


class ReferenceFollower:
    """
    Follows reference chains using Depth-First Search (DFS).

    Strategy:
    - Start from seed paper
    - Extract references
    - For each reference, recursively follow (up to max_depth)
    - Stop when max_papers reached or max_depth exceeded
    - Track visited papers to avoid cycles
    """

    def __init__(self, max_depth: int = 3, max_papers: int = 5):
        """
        Initialize reference follower.

        Args:
            max_depth: Maximum depth to follow references (default: 3 hops)
            max_papers: Maximum total papers to collect (default: 5)
        """
        self.max_depth = max_depth
        self.max_papers = max_papers
        self.visited: Set[str] = set()  # Track visited paper IDs
        self.papers_collected: List[Dict] = []
        self.reference_chains: List[List[str]] = []  # Track reference paths
        self.paper_reader = PaperReader()

    async def follow_dfs(
        self,
        seed_paper: Dict,
        current_depth: int = 0,
        current_chain: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Follow references from seed paper using DFS.

        Args:
            seed_paper: Starting paper dict with at least 'title' and 'pdf_url'
            current_depth: Current recursion depth
            current_chain: Current reference chain (for tracking)

        Returns:
            List of papers collected during DFS
        """
        if current_chain is None:
            current_chain = []

        # Stop conditions
        if current_depth >= self.max_depth:
            return self.papers_collected

        if len(self.papers_collected) >= self.max_papers:
            return self.papers_collected

        # Get paper identifier (use title as ID for now)
        paper_id = seed_paper.get("title", "")

        # Skip if already visited
        if paper_id in self.visited:
            return self.papers_collected

        self.visited.add(paper_id)

        # Update chain
        current_chain = current_chain + [paper_id]

        # Try to read PDF if available
        pdf_url = seed_paper.get("pdf_url")
        if pdf_url and len(self.papers_collected) < self.max_papers:
            print(f"   🔍 DFS depth {current_depth}: {paper_id[:60]}...")

            pdf_content = await self.paper_reader.read_pdf(pdf_url)

            if pdf_content and pdf_content.get("success"):
                # Add full text to paper
                paper_with_fulltext = seed_paper.copy()
                paper_with_fulltext["full_text"] = pdf_content["full_text"]
                paper_with_fulltext["sections"] = pdf_content["sections"]

                self.papers_collected.append(paper_with_fulltext)

                # Extract references
                references = self.paper_reader.extract_references(pdf_content)

                # Follow references recursively (DFS)
                for ref in references:
                    if len(self.papers_collected) >= self.max_papers:
                        break

                    # Recursively follow this reference
                    await self.follow_dfs(
                        seed_paper=ref,
                        current_depth=current_depth + 1,
                        current_chain=current_chain
                    )

                # Record this chain
                if len(current_chain) > 1:
                    self.reference_chains.append(current_chain)

        return self.papers_collected

    def get_chains(self) -> List[List[str]]:
        """
        Get all reference chains discovered during DFS.

        Returns:
            List of chains, where each chain is a list of paper titles
        """
        return self.reference_chains

    def reset(self):
        """Reset the follower for a new DFS run."""
        self.visited.clear()
        self.papers_collected = []
        self.reference_chains = []


class PaperSearchRefiner:
    """
    Refines search terms based on seed papers.

    Used by character agents to improve search quality after reviewing
    seed papers from broad search.
    """

    @staticmethod
    async def refine_search_terms(
        original_topic: str,
        seed_papers: List[Dict],
        llm_model
    ) -> str:
        """
        Refine search terms based on seed paper analysis.

        Args:
            original_topic: Original topic/query string
            seed_papers: List of seed paper dicts (with title/abstract)
            llm_model: LLM model to use for refinement

        Returns:
            Refined search query string
        """
        # Prepare seed paper summaries
        summaries = []
        for paper in seed_papers[:5]:  # Max 5 papers
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")[:300]  # Truncate abstract
            summaries.append(f"- {title}\n  {abstract}")

        summaries_str = "\n".join(summaries)

        prompt = f"""
Based on the research topic and these seed papers, suggest a refined search query
that would find more relevant papers.

Original Topic: {original_topic}

Seed Papers:
{summaries_str}

Provide a concise, focused search query (1-2 sentences) that captures the key concepts
and would improve search results.
"""

        # Call LLM
        try:
            response = await llm_model.ainvoke(prompt)
            refined_query = response.content.strip()

            print(f"   🔍 Refined search query: {refined_query}")

            return refined_query
        except Exception as e:
            print(f"   ⚠️  Error refining search: {e}")
            return original_topic  # Fallback to original


def select_papers_for_deep_dive(
    seed_papers: List[Dict],
    max_papers: int = 5,
    criteria: str = "citations"
) -> List[Dict]:
    """
    Select papers for deep dive (PDF reading + reference following).

    Args:
        seed_papers: List of seed papers from broad search
        max_papers: Maximum papers to select
        criteria: Selection criteria ("citations", "recency", "relevance")

    Returns:
        Subset of papers selected for deep dive
    """
    if not seed_papers:
        return []

    # Filter papers with PDF URLs
    papers_with_pdf = [p for p in seed_papers if p.get("pdf_url")]

    if not papers_with_pdf:
        print("   ⚠️  No papers with PDF URLs available")
        return []

    # Sort by criteria
    if criteria == "citations":
        papers_with_pdf.sort(key=lambda x: x.get("citations", 0), reverse=True)
    elif criteria == "recency":
        papers_with_pdf.sort(key=lambda x: x.get("year", 0), reverse=True)
    elif criteria == "relevance":
        papers_with_pdf.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    # Select top N
    selected = papers_with_pdf[:max_papers]

    print(f"   📌 Selected {len(selected)} papers for deep dive (from {len(seed_papers)} seed papers)")

    return selected


# Example usage (for testing)
async def example_usage():
    """Example of how to use the paper reading module."""

    # Example seed paper
    seed_paper = {
        "title": "Attention Is All You Need",
        "authors": ["Vaswani et al."],
        "abstract": "The dominant sequence transduction models...",
        "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf",
        "year": 2017,
        "citations": 50000
    }

    # 1. Read PDF
    reader = PaperReader()
    pdf_content = await reader.read_pdf(seed_paper["pdf_url"])
    print(f"Read PDF: {pdf_content['success']}")

    # 2. Follow references
    follower = ReferenceFollower(max_depth=3, max_papers=5)
    papers = await follower.follow_dfs(seed_paper)
    print(f"Collected {len(papers)} papers via DFS")

    # 3. Get reference chains
    chains = follower.get_chains()
    print(f"Discovered {len(chains)} reference chains")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
