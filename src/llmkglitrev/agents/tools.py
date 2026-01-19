"""Research Utilities and Tools.

This module provides search and content processing utilities for the research agent,
including web search capabilities and content summarization tools.
"""

from pathlib import Path
from datetime import datetime
from typing_extensions import Annotated, List, Literal
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from tavily import TavilyClient
from pydantic import BaseModel, Field
from llmkglitrev.agents.states import ResearchSummary
from llmkglitrev.agents.prompts.research_summary import summarize_webpage_prompt

# Import docling for HTML to markdown conversion
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("Warning: docling not available. Install with: pip install docling")

# from deep_research_from_scratch.state_research import Summary
# from deep_research_from_scratch.prompts import summarize_webpage_prompt

# ===== UTILITY FUNCTIONS =====

# Use GPT-4o-mini for cost-effective summarization
# summarization_model = init_chat_model(model="openai:gpt-4o-mini")
summarization_model = init_chat_model(model="deepseek:deepseek-chat")
tavily_client = TavilyClient()

# Academic domains to prioritize in search
ACADEMIC_DOMAINS = [
    "ieee.org",
    "springer.com",
    "acm.org",
    "scholar.google.com",
    "arxiv.org",
    "sciencedirect.com",
    "nature.com",
    "science.org",
    "plos.org",
    "bmj.com",
    "thelancet.com",
    "cell.com",
    "wiley.com",
    "tandfonline.com",
    "cambridge.org",
    "oxford.org",
    "jstor.org",
    "researchgate.net",
    "semanticscholar.org",
    "pubmed.ncbi.nlm.nih.gov",
]


def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")


def convert_html_to_markdown(html_content: str) -> str:
    """Convert HTML content to markdown format using docling.

    Args:
        html_content: Raw HTML content from webpage

    Returns:
        Markdown-formatted content or original HTML if conversion fails
    """
    if not DOCLING_AVAILABLE:
        # Fallback: return raw content if docling not available
        return html_content

    try:
        # Initialize document converter
        converter = DocumentConverter()

        # Convert HTML to markdown
        # Note: docling works with file paths or URLs
        # For raw HTML content, we need to save temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_content)
            temp_path = f.name

        try:
            # Convert document
            result = converter.convert(temp_path)
            markdown_content = result.document.export_to_markdown()
            return markdown_content
        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    except Exception as e:
        print(f"Failed to convert HTML to markdown: {str(e)}")
        return html_content


def is_academic_source(url: str) -> bool:
    """Check if URL is from an academic source.

    Args:
        url: URL to check

    Returns:
        True if URL contains academic domain, False otherwise
    """
    url_lower = url.lower()
    return any(domain in url_lower for domain in ACADEMIC_DOMAINS)

def format_search_output(summarized_results: dict) -> str:
    """Format search results into a well-structured string output.

    Args:
        summarized_results: Dictionary of processed search results

    Returns:
        Formatted string of search results with clear source separation
    """
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    # Separate academic and non-academic sources
    academic_results = {url: result for url, result in summarized_results.items()
                       if is_academic_source(url)}
    non_academic_results = {url: result for url, result in summarized_results.items()
                           if not is_academic_source(url)}

    formatted_output = "Search results:\n\n"

    # Prioritize academic sources first
    if academic_results:
        formatted_output += "=== ACADEMIC SOURCES ===\n\n"
        for i, (url, result) in enumerate(academic_results.items(), 1):
            formatted_output += f"\n--- SOURCE {i}: {result['title']} [ACADEMIC] ---\n"
            formatted_output += f"URL: {url}\n\n"
            formatted_output += f"SUMMARY:\n{result['content']}\n\n"
            formatted_output += "-" * 80 + "\n"

    # Then show non-academic sources
    if non_academic_results:
        formatted_output += "\n=== OTHER SOURCES ===\n\n"
        start_idx = len(academic_results) + 1
        for i, (url, result) in enumerate(non_academic_results.items(), start_idx):
            formatted_output += f"\n--- SOURCE {i}: {result['title']} ---\n"
            formatted_output += f"URL: {url}\n\n"
            formatted_output += f"SUMMARY:\n{result['content']}\n\n"
            formatted_output += "-" * 80 + "\n"

    return formatted_output


def tavily_search_multiple(
    search_queries: List[str],
    max_results: int = 3,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
    prioritize_academic: bool = True,
) -> List[dict]:
    """Perform search using Tavily API for multiple queries with academic source priority.

    Args:
        search_queries: List of search queries to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content
        prioritize_academic: Whether to prioritize academic sources in results

    Returns:
        List of search result dictionaries
    """

    # Execute searches sequentially. Note: you can use AsyncTavilyClient to parallelize this step.
    search_docs = []
    for query in search_queries:
        # Add academic keywords to encourage scholarly results
        enhanced_query = query
        if prioritize_academic:
            # Add academic context to query
            enhanced_query = f"{query} site:scholar.google.com OR site:ieee.org OR site:arxiv.org OR site:springer.com OR site:acm.org"

        try:
            result = tavily_client.search(
                enhanced_query,
                max_results=max_results * 2,  # Request more to filter for academic
                include_raw_content=include_raw_content,
                topic=topic
            )

            # Filter and prioritize academic sources
            if prioritize_academic and 'results' in result:
                academic_results = [r for r in result['results'] if is_academic_source(r.get('url', ''))]
                other_results = [r for r in result['results'] if not is_academic_source(r.get('url', ''))]

                # Combine: academic first, then others up to max_results
                filtered_results = academic_results[:max_results]
                remaining_slots = max_results - len(filtered_results)
                if remaining_slots > 0:
                    filtered_results.extend(other_results[:remaining_slots])

                result['results'] = filtered_results

            search_docs.append(result)

        except Exception as e:
            print(f"Search error for query '{query}': {str(e)}")
            # Return empty result for failed query
            search_docs.append({'results': []})

    return search_docs


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Fetch results from Tavily search API with content summarization.

    Args:
        query: A single search query to execute
        max_results: Maximum number of results to return
        topic: Topic to filter results by ('general', 'news', 'finance')

    Returns:
        Formatted string of search results with summaries
    """
    # Execute search for single query
    search_results = tavily_search_multiple(
        [query],  # Convert single query to list for the internal function
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    )

    # Deduplicate results by URL to avoid processing duplicate content
    unique_results = deduplicate_search_results(search_results)

    # Process results with summarization
    summarized_results = process_search_results(unique_results)

    # Format output for consumption
    return format_search_output(summarized_results)

def summarize_webpage_content(webpage_content: str, convert_to_markdown: bool = True, max_chars: int = 60000) -> str:
    """Summarize webpage content using the configured summarization model.

    Args:
        webpage_content: Raw webpage content to summarize
        convert_to_markdown: Whether to convert HTML to markdown first (saves context)
        max_chars: Maximum characters to process (default: 60000)

    Returns:
        Formatted summary with key excerpts
    """
    try:
        # Limit content length to avoid excessive processing
        if len(webpage_content) > max_chars:
            print(f"⚠️  Content too long ({len(webpage_content)} chars), truncating to {max_chars} chars")
            webpage_content = webpage_content[:max_chars]

        # Convert HTML to markdown for more concise context
        content_to_summarize = webpage_content
        if convert_to_markdown and DOCLING_AVAILABLE:
            try:
                content_to_summarize = convert_html_to_markdown(webpage_content)
                print(f"✓ Converted HTML to markdown (reduced from {len(webpage_content)} to {len(content_to_summarize)} chars)")

                # Apply limit again after conversion if needed
                if len(content_to_summarize) > max_chars:
                    print(f"⚠️  Markdown still too long ({len(content_to_summarize)} chars), truncating to {max_chars} chars")
                    content_to_summarize = content_to_summarize[:max_chars]
            except Exception as e:
                print(f"Markdown conversion failed, using raw content: {str(e)}")
                content_to_summarize = webpage_content

        # Set up structured output model for summarization
        structured_model = summarization_model.with_structured_output(ResearchSummary)

        # Generate summary from markdown content
        summary = structured_model.invoke([
            HumanMessage(content=summarize_webpage_prompt.format(
                webpage_content=content_to_summarize,
                date=get_today_str()
            ))
        ])

        # Format summary with clear structure
        formatted_summary = (
            f"<topic>\n{summary.topic}\n</topic>\n\n"
            f"<research_question>\n{summary.research_question}\n</research_question>\n\n"
            f"<method>\n{summary.method}\n</method>\n\n"
            f"<result>\n{summary.result}\n</result>\n"
        )

        return formatted_summary

    except Exception as e:
        print(f"Failed to summarize webpage: {str(e)}")
        return webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content




def deduplicate_search_results(search_results: List[dict]) -> dict:
    """Deduplicate search results by URL to avoid processing duplicate content.
    
    Args:
        search_results: List of search result dictionaries
        
    Returns:
        Dictionary mapping URLs to unique results
    """
    unique_results = {}
    
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result
    
    return unique_results

def process_search_results(unique_results: dict) -> dict:
    """Process search results by summarizing content where available.
    
    Args:
        unique_results: Dictionary of unique search results
        
    Returns:
        Dictionary of processed results with summaries
    """
    summarized_results = {}
    
    for url, result in unique_results.items():
        # Use existing content if no raw content for summarization
        if not result.get("raw_content"):
            content = result['content']
        else:
            # Summarize raw content for better processing
            content = summarize_webpage_content(result['raw_content'])
        
        summarized_results[url] = {
            'title': result['title'],
            'content': content
        }
    
    return summarized_results




@tool(parse_docstring=True)
def planning_tool(planning_reflection: str) -> str:
    """   
    Tool for strategic evaluation on search.
    
    Use this tool if you think that you can propose a research agenda that built on top of current literature but suitable to and not too deviated from researcher expertise.
    Draft a research plan and compare to current literature. Check if your derived research plan is built upon research gap of the current literature and suitable to researcher profile.
    Use this tool strategically to see if you need to search more literature or current plan is good enough.
    
    When to use:
    - After evaluating current literature: Is that research plan intresting and feasible?
    
    Args:
        planning_reflection: Your detailed research plan.         
    """
    return f"Planning Reflection recorded: {planning_reflection}"

@tool(parse_docstring=True)
def evaluation_tool(evaluation_reflection: str) -> str:
    """
    Tool for strategic reflection on research progress and decision-making.
    
    Use this tool if you are confident that you can propose a research agenda that built on top of current literature but suitable to and not too deviated from researcher expertise.
    This creates a deliberate pause in the research workflow for quality decision-making.
    
    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to comprehensive knowledge about the literature and what to search next?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Do you find the niche that I can contribute to the scientific development?
    
    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have enough knowledge about the current literature and identify gap or novelty that I can contribute?
    4. Strategic decision - Should I continue searching or provide my answer?
    
    Args:
        evaluation_reflection: Your detailed reflection on research progress, findings, gaps, and next steps
    """
    return f"Evaluation Reflection recorded: {evaluation_reflection}"

@tool(parse_docstring=True)
def supervisor_think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find? Should I investigate more 
    - Before deciding next steps: What aspects should I look for research ideas, based on research profiles and potential direction? 
    - When assessing research gaps: Is this research gap novel enough but still close to resesearcher profile to make it plausible and feasible?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - Is the gap novel and close to researcher profile and expertise?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"



@tool
class ConductResearch(BaseModel):
    """Tool for delegating a research task to a specialized sub-agent."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

@tool
class ResearchComplete(BaseModel):
    """Tool for indicating that the research process is complete."""
    pass

# @tool
# class AskHumanFeedback(BaseModel):
#     """Tool to ask human if they are satisfied with the current directions to dived in more."""
#     pass