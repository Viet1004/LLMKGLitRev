"""
Neo4j vector search for literature retrieval
"""
from typing import List, Dict
from langchain_openai.embeddings import OpenAIEmbeddings
from neo4j import GraphDatabase
import os


class Neo4jLiteratureSearch:
    """Search papers in Neo4j using vector similarity"""
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "123456789",
        embedding_model: str = "text-embedding-3-small"
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embedding_model = OpenAIEmbeddings(model=embedding_model)
    
    def search_similar_papers(
        self, 
        query_text: str, 
        top_k: int = 7
    ) -> List[Dict]:
        """
        Search for papers similar to query text.
        
        Args:
            query_text: Natural language query
            top_k: Number of papers to return
            
        Returns:
            List of paper dictionaries with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query_text)
            
            # Search Neo4j using vector similarity
            with self.driver.session() as session:
                # Cypher query for vector similarity search
                cypher_query = """
                MATCH (p:Paper)
                WITH p, 
                     gds.similarity.cosine(p.embedding, $query_embedding) AS similarity
                WHERE similarity > 0.5
                RETURN p.title AS title,
                       p.authors AS authors,
                       p.abstract AS abstract,
                       p.year AS year,
                       p.venue AS venue,
                       p.url AS url,
                       similarity
                ORDER BY similarity DESC
                LIMIT $top_k
                """
                
                result = session.run(
                    cypher_query,
                    query_embedding=query_embedding,
                    top_k=top_k
                )
                
                papers = []
                for record in result:
                    papers.append({
                        "title": record["title"],
                        "authors": record["authors"],
                        "abstract": record["abstract"],
                        "year": record["year"],
                        "venue": record["venue"],
                        "url": record["url"],
                        "similarity": round(record["similarity"], 3)
                    })
                
                return papers
                
        except Exception as e:
            print(f"Error searching Neo4j: {e}")
            return []
    
    def format_papers_for_llm(self, papers: List[Dict]) -> str:
        """
        Format retrieved papers for LLM context.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Formatted string for LLM prompt
        """
        if not papers:
            return "No relevant papers found in literature database."
        
        formatted = f"## Retrieved Literature ({len(papers)} papers from knowledge base):\n\n"
        
        for i, paper in enumerate(papers, 1):
            formatted += f"**[{i}] {paper['title']}**\n"
            formatted += f"   Authors: {paper['authors']}\n"
            formatted += f"   Year: {paper['year']}\n"
            
            # Truncate long abstracts
            abstract = paper['abstract']
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."
            formatted += f"   Abstract: {abstract}\n"
            formatted += f"   Relevance Score: {paper['similarity']}\n"
            formatted += f"   URL: {paper.get('url', 'N/A')}\n\n"
        
        return formatted
    
    def close(self):
        """Close Neo4j connection"""
        self.driver.close()


# Singleton instance
_neo4j_search = None

def get_neo4j_search() -> Neo4jLiteratureSearch:
    """Get or create Neo4j search instance"""
    global _neo4j_search
    if _neo4j_search is None:
        _neo4j_search = Neo4jLiteratureSearch()
    return _neo4j_search
