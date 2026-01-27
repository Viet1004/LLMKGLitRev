"""
Domain Ontology Manager for enhancing research agents with domain knowledge.

This module provides tools to:
1. Load domain-specific ontologies
2. Query relationships between concepts
3. Validate terminology in research papers
4. Generate competency questions for Socratic dialogue
"""

from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
import re

from owlready2 import *


class DomainOntologyManager:
    """
    Manage domain-specific ontologies for research characters.

    This class provides an interface to load and query ontologies,
    helping research agents:
    - Validate terminology used in papers
    - Identify related concepts not mentioned
    - Detect terminology mismatches
    - Generate ontology-grounded questions
    """

    # Pre-configured ontology sources
    ONTOLOGY_SOURCES = {
        "edam": {
            "url": "http://edamontology.org/EDAM.owl",
            "name": "EDAM - Data Analysis and Management Ontology",
            "domains": ["data science", "machine learning", "bioinformatics"],
            "description": "Ontology for data analysis, ML, and bioimaging"
        },
        "ml-schema": {
            "url": "http://www.w3.org/ns/mls",
            "name": "ML-Schema (W3C)",
            "domains": ["machine learning", "data mining"],
            "description": "W3C standard for ML algorithms and datasets"
        }
    }

    def __init__(
        self,
        ontology_source: str = "edam",
        custom_url: Optional[str] = None
    ):
        """
        Initialize ontology manager.

        Args:
            ontology_source: Name of pre-configured ontology or "custom"
            custom_url: URL to custom ontology (if ontology_source="custom")
        """
        self.ontology_source = ontology_source
        self.ontology = None
        self.graph = None
        self._classes_cache = None
        self._term_index = None

        if custom_url:
            self.ontology_url = custom_url
            self.ontology_name = "Custom Ontology"
        elif ontology_source in self.ONTOLOGY_SOURCES:
            config = self.ONTOLOGY_SOURCES[ontology_source]
            self.ontology_url = config["url"]
            self.ontology_name = config["name"]
        else:
            raise ValueError(
                f"Unknown ontology source '{ontology_source}'. "
                f"Available: {list(self.ONTOLOGY_SOURCES.keys())} or 'custom'"
            )

    def load(self) -> bool:
        """
        Load the ontology from source.

        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"📥 Loading {self.ontology_name} from {self.ontology_url}...")

            self.ontology = get_ontology(self.ontology_url).load()

            # Get RDFlib graph for SPARQL queries
            try:
                self.graph = default_world.as_rdflib_graph()
            except:
                print("⚠️  RDFlib graph creation failed, SPARQL queries unavailable")
                self.graph = None

            # Build indexes for fast lookup
            self._build_term_index()

            print(f"✅ Loaded {self.ontology_name} successfully!")
            return True

        except Exception as e:
            print(f"❌ Failed to load ontology: {e}")
            return False

    def _build_term_index(self):
        """Build an index of all terms (class names, labels) for fast search."""
        self._term_index = {}
        self._classes_cache = list(self.ontology.classes())

        for cls in self._classes_cache:
            # Index by class name
            name_lower = str(cls.name).lower()
            if name_lower not in self._term_index:
                self._term_index[name_lower] = []
            self._term_index[name_lower].append(cls)

            # Index by labels if available
            if hasattr(cls, 'label') and cls.label:
                labels = cls.label if isinstance(cls.label, list) else [cls.label]
                for label in labels:
                    label_lower = str(label).lower()
                    if label_lower not in self._term_index:
                        self._term_index[label_lower] = []
                    self._term_index[label_lower].append(cls)

    def find_concept(self, term: str) -> Optional[ThingClass]:
        """
        Find a concept in the ontology by name or label.

        Args:
            term: The term to search for

        Returns:
            The matching class, or None if not found
        """
        if not self._term_index:
            return None

        term_lower = term.lower()

        # Exact match
        if term_lower in self._term_index:
            return self._term_index[term_lower][0]

        # Partial match
        for key, classes in self._term_index.items():
            if term_lower in key or key in term_lower:
                return classes[0]

        return None

    def get_concept_label(self, concept_name: str) -> str:
        """
        Get human-readable label for a concept.

        Args:
            concept_name: The concept class name (e.g., 'data_id')

        Returns:
            Human-readable label (e.g., 'Data identifier') or the original name if no label found
        """
        concept = self.find_concept(concept_name)

        if not concept:
            # Fallback: convert underscores to spaces and title case
            return concept_name.replace('_', ' ').title()

        # Try to get rdfs:label
        if hasattr(concept, 'label') and concept.label:
            # Labels are stored as list in owlready2
            if isinstance(concept.label, list) and len(concept.label) > 0:
                return concept.label[0]
            elif isinstance(concept.label, str):
                return concept.label

        # Fallback: use the name with formatting
        return concept_name.replace('_', ' ').title()

    def find_related_concepts(
        self,
        term: str,
        max_results: int = 10
    ) -> List[Dict[str, any]]:
        """
        Find concepts related to a given term.

        This includes:
        - Direct matches
        - Superclasses
        - Subclasses
        - Concepts with similar names

        Args:
            term: The term to search for
            max_results: Maximum number of results to return

        Returns:
            List of dicts with concept information
        """
        results = []

        # Find the primary concept
        primary_concept = self.find_concept(term)

        if primary_concept:
            # Add the primary concept
            results.append({
                "name": primary_concept.name,
                "type": "exact_match",
                "definition": self._get_definition(primary_concept),
                "class": primary_concept
            })

            # Add superclasses
            supers = [s for s in primary_concept.is_a
                     if isinstance(s, ThingClass) and s.name != 'Thing']
            for sup in supers[:3]:
                results.append({
                    "name": sup.name,
                    "type": "superclass",
                    "definition": self._get_definition(sup),
                    "class": sup
                })

            # Add subclasses
            subs = list(primary_concept.subclasses())
            for sub in subs[:3]:
                results.append({
                    "name": sub.name,
                    "type": "subclass",
                    "definition": self._get_definition(sub),
                    "class": sub
                })

        # Add similar concepts
        if self._term_index:
            term_words = set(term.lower().split())
            for key, classes in self._term_index.items():
                key_words = set(key.split())
                # Check for word overlap
                if term_words & key_words and len(results) < max_results:
                    for cls in classes[:1]:  # One per key
                        if not any(r["name"] == cls.name for r in results):
                            results.append({
                                "name": cls.name,
                                "type": "similar",
                                "definition": self._get_definition(cls),
                                "class": cls
                            })

        return results[:max_results]

    def _get_definition(self, cls: ThingClass) -> str:
        """Extract definition/comment from a class."""
        if hasattr(cls, 'comment') and cls.comment:
            comment = cls.comment[0] if isinstance(cls.comment, list) else cls.comment
            return str(comment)

        if hasattr(cls, 'label') and cls.label:
            label = cls.label[0] if isinstance(cls.label, list) else cls.label
            return str(label)

        return "No definition available"

    def validate_terminology(self, text: str) -> List[Dict]:
        """
        Check if terms in text match ontology concepts.

        Extracts potential domain terms from text and checks
        if they exist in the ontology.

        Args:
            text: Text to analyze

        Returns:
            List of validation results
        """
        results = []

        # Extract potential terms (simple heuristic: capitalized words, technical terms)
        words = text.split()

        # Look for multi-word terms and single technical words
        potential_terms = []

        # Extract capitalized phrases
        current_phrase = []
        for word in words:
            clean_word = re.sub(r'[^\w\s-]', '', word)
            if clean_word and (clean_word[0].isupper() or '-' in clean_word):
                current_phrase.append(clean_word)
            else:
                if current_phrase:
                    potential_terms.append(' '.join(current_phrase))
                    current_phrase = []
        if current_phrase:
            potential_terms.append(' '.join(current_phrase))

        # Check each term against ontology
        for term in potential_terms:
            concept = self.find_concept(term)
            if concept:
                results.append({
                    "term": term,
                    "status": "found",
                    "ontology_concept": concept.name,
                    "definition": self._get_definition(concept)
                })
            else:
                results.append({
                    "term": term,
                    "status": "not_found",
                    "ontology_concept": None,
                    "definition": None
                })

        return results

    def get_missing_relationships(
        self,
        terms: List[str]
    ) -> List[Dict]:
        """
        Identify relationships between terms that exist in the ontology
        but are not explicitly discussed.

        Args:
            terms: List of terms mentioned in text

        Returns:
            List of missing relationships
        """
        missing_relationships = []

        # Find concepts for each term
        concepts = {}
        for term in terms:
            concept = self.find_concept(term)
            if concept:
                concepts[term] = concept

        # Check for relationships between concepts
        concept_list = list(concepts.values())
        for i, concept1 in enumerate(concept_list):
            for concept2 in concept_list[i+1:]:
                # Check if concept1 is related to concept2
                relationship = self._find_relationship(concept1, concept2)
                if relationship:
                    missing_relationships.append({
                        "concept1": concept1.name,
                        "concept2": concept2.name,
                        "relationship": relationship
                    })

        return missing_relationships

    def _find_relationship(
        self,
        concept1: ThingClass,
        concept2: ThingClass
    ) -> Optional[str]:
        """
        Check if two concepts have a relationship in the ontology.

        Returns:
            Description of relationship, or None
        """
        # Check if one is subclass of the other
        if concept2 in concept1.is_a:
            return f"{concept1.name} is a subclass of {concept2.name}"
        if concept1 in concept2.is_a:
            return f"{concept2.name} is a subclass of {concept1.name}"

        # Check if they share a common parent
        concept1_parents = set(c for c in concept1.is_a if isinstance(c, ThingClass))
        concept2_parents = set(c for c in concept2.is_a if isinstance(c, ThingClass))
        common_parents = concept1_parents & concept2_parents
        if common_parents:
            parent = list(common_parents)[0]
            return f"Both are types of {parent.name}"

        return None

    def generate_competency_questions(
        self,
        research_text: str,
        priority_threshold: int = 7
    ) -> List[Dict]:
        """
        Generate Socratic questions based on ontology analysis.

        This analyzes the research text and generates questions about:
        1. Missing related concepts
        2. Terminology clarifications
        3. Unexplored relationships
        4. Alternative approaches

        Args:
            research_text: The research output text to analyze
            priority_threshold: Minimum priority for questions (1-10)

        Returns:
            List of competency questions with metadata
        """
        questions = []

        # Validate terminology
        validation_results = self.validate_terminology(research_text)

        # Extract found terms
        found_terms = [r["term"] for r in validation_results if r["status"] == "found"]

        # For each found term, suggest related concepts
        for term in found_terms:
            related = self.find_related_concepts(term, max_results=5)

            # Filter to concepts not mentioned in text
            for concept_info in related:
                concept_name = concept_info["name"]

                # Skip if already mentioned
                if concept_name.lower() in research_text.lower():
                    continue

                # Generate question based on relationship type
                if concept_info["type"] == "superclass":
                    question = {
                        "question": f"How does this work relate to the broader concept of {concept_name}?",
                        "concept": concept_name,
                        "definition": concept_info["definition"],
                        "type": "broader_context",
                        "priority": 7
                    }
                    questions.append(question)

                elif concept_info["type"] == "subclass":
                    question = {
                        "question": f"Have you considered the specific approach of {concept_name}?",
                        "concept": concept_name,
                        "definition": concept_info["definition"],
                        "type": "alternative_approach",
                        "priority": 8
                    }
                    questions.append(question)

                elif concept_info["type"] == "similar":
                    question = {
                        "question": f"How does '{term}' compare to the related concept '{concept_name}'?",
                        "concept": concept_name,
                        "definition": concept_info["definition"],
                        "type": "comparison",
                        "priority": 6
                    }
                    questions.append(question)

        # Check for missing relationships
        missing_rels = self.get_missing_relationships(found_terms)
        for rel in missing_rels:
            question = {
                "question": f"The ontology indicates that {rel['relationship']}. How does this affect your research?",
                "concept": f"{rel['concept1']} and {rel['concept2']}",
                "definition": rel['relationship'],
                "type": "relationship_clarification",
                "priority": 9
            }
            questions.append(question)

        # Filter by priority
        questions = [q for q in questions if q["priority"] >= priority_threshold]

        # Sort by priority (highest first)
        questions.sort(key=lambda x: x["priority"], reverse=True)

        return questions

    def get_statistics(self) -> Dict:
        """
        Get statistics about the loaded ontology.

        Returns:
            Dictionary with ontology statistics
        """
        if not self.ontology:
            return {"error": "Ontology not loaded"}

        classes = list(self.ontology.classes())
        object_props = list(self.ontology.object_properties())
        data_props = list(self.ontology.data_properties())

        return {
            "name": self.ontology_name,
            "url": self.ontology_url,
            "total_classes": len(classes),
            "object_properties": len(object_props),
            "data_properties": len(data_props),
            "indexed_terms": len(self._term_index) if self._term_index else 0
        }

    def extract_concepts(self, text: str) -> List[str]:
        """Extract ontology concepts mentioned in text.

        Args:
            text: Text to analyze

        Returns:
            List of concept names found in text
        """
        if not self._term_index:
            return []

        concepts_found = []
        text_lower = text.lower()

        # Check each indexed term against the text
        for term_key in self._term_index.keys():
            # Use word boundaries to avoid partial matches
            if re.search(r'\b' + re.escape(term_key) + r'\b', text_lower):
                # Get the actual concept name (not the lowercased key)
                concept = self._term_index[term_key][0]
                concepts_found.append(concept.name)

        return list(set(concepts_found))  # Deduplicate

    def get_related_concepts(
        self,
        concepts: List[str],
        max_per_concept: int = 3
    ) -> List[str]:
        """Get concepts related to the given concepts.

        Args:
            concepts: List of concept names
            max_per_concept: Maximum related concepts per input concept

        Returns:
            List of related concept names
        """
        related = []

        for concept_name in concepts:
            concept = self.find_concept(concept_name)
            if concept:
                # Get superclasses
                supers = [s.name for s in concept.is_a
                         if isinstance(s, ThingClass) and s.name != 'Thing']
                related.extend(supers[:max_per_concept])

                # Get subclasses
                subs = [s.name for s in concept.subclasses()]
                related.extend(subs[:max_per_concept])

        return list(set(related))  # Deduplicate

    def get_concept_relationships(
        self,
        concepts: List[str]
    ) -> List[Dict[str, str]]:
        """Get relationships between concepts.

        Args:
            concepts: List of concept names

        Returns:
            List of relationship dicts with source, target, relationship_type
        """
        relationships = []

        # Map concept names to classes
        concept_map = {}
        for concept_name in concepts:
            concept = self.find_concept(concept_name)
            if concept:
                concept_map[concept_name] = concept

        # Find relationships between all pairs
        concept_names = list(concept_map.keys())
        for i, name1 in enumerate(concept_names):
            for name2 in concept_names[i+1:]:
                concept1 = concept_map[name1]
                concept2 = concept_map[name2]

                # Check if one is subclass of the other
                if concept2 in concept1.is_a:
                    relationships.append({
                        "source": name1,
                        "target": name2,
                        "relationship": "is_a"
                    })
                elif concept1 in concept2.is_a:
                    relationships.append({
                        "source": name2,
                        "target": name1,
                        "relationship": "is_a"
                    })
                else:
                    # Check for common parent
                    concept1_parents = set(c.name for c in concept1.is_a
                                          if isinstance(c, ThingClass) and c.name != 'Thing')
                    concept2_parents = set(c.name for c in concept2.is_a
                                          if isinstance(c, ThingClass) and c.name != 'Thing')
                    common_parents = concept1_parents & concept2_parents

                    if common_parents:
                        parent = list(common_parents)[0]
                        relationships.append({
                            "source": name1,
                            "target": name2,
                            "relationship": f"related_via_{parent}"
                        })

        return relationships

    def get_concept_definition(self, concept_name: str) -> str:
        """Get definition for a concept.

        Args:
            concept_name: Name of the concept

        Returns:
            Definition string or "Not found"
        """
        concept = self.find_concept(concept_name)
        if concept:
            return self._get_definition(concept)
        return "Concept not found in ontology"

    def search_similar_concepts(
        self,
        query: str,
        similarity_threshold: float = 0.7,
        max_results: int = 10
    ) -> List[Dict[str, any]]:
        """Search for concepts similar to query string.

        Args:
            query: Search query
            similarity_threshold: Minimum similarity score (0-1)
            max_results: Maximum number of results

        Returns:
            List of similar concepts with similarity scores
        """
        if not self._term_index:
            return []

        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for term_key, classes in self._term_index.items():
            term_words = set(term_key.split())

            # Calculate simple word overlap similarity
            if query_words and term_words:
                overlap = len(query_words & term_words)
                similarity = overlap / max(len(query_words), len(term_words))

                if similarity >= similarity_threshold:
                    concept = classes[0]
                    results.append({
                        "name": concept.name,
                        "definition": self._get_definition(concept),
                        "similarity": similarity,
                        "matched_term": term_key
                    })

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results[:max_results]

    def build_concept_tree(self, concepts: List[str]) -> Dict:
        """
        Build hierarchical tree structure from flat list of concepts.

        Args:
            concepts: List of concept names to build tree from

        Returns:
            Tree structure with levels (0=root, 1=general, 2=specific, ...)
        """
        if not self.ontology:
            return {}

        # Find root concepts (those with no parents in the concept list)
        roots = []

        for concept in concepts:
            parents = self._get_parent_concepts(concept)
            has_parent_in_list = any(p in concepts for p in parents)

            if not has_parent_in_list:
                roots.append(concept)

        # Build tree recursively
        tree = {
            "name": "Research Concepts",
            "children": [self._build_subtree(root, concepts, level=0) for root in roots]
        }

        return tree

    def _build_subtree(self, concept: str, all_concepts: List[str], level: int) -> Dict:
        """Recursively build subtree for a concept."""
        children = self._get_child_concepts(concept)
        child_nodes = [
            self._build_subtree(child, all_concepts, level + 1)
            for child in children
            if child in all_concepts
        ]

        return {
            "name": concept,
            "level": level,
            "children": child_nodes
        }

    def _get_parent_concepts(self, concept_name: str) -> List[str]:
        """Get parent concepts (superclasses) for a concept."""
        concept = self.find_concept(concept_name)
        if not concept:
            return []

        from owlready2 import ThingClass
        parents = [
            p.name for p in concept.is_a
            if isinstance(p, ThingClass) and p.name != 'Thing'
        ]
        return parents

    def _get_child_concepts(self, concept_name: str) -> List[str]:
        """Get child concepts (subclasses) for a concept."""
        concept = self.find_concept(concept_name)
        if not concept:
            return []

        children = [c.name for c in concept.subclasses()]
        return children

    def get_top_level_parent(self, concept_name: str) -> str:
        """
        Get the top-level parent for clustering.

        Args:
            concept_name: Concept to find top parent for

        Returns:
            Name of top-level parent concept
        """
        parents = self._get_parent_concepts(concept_name)

        if not parents:
            return concept_name

        # Recursively find top parent
        for parent in parents:
            grandparents = self._get_parent_concepts(parent)
            if grandparents:
                return self.get_top_level_parent(parent)

        return parents[0] if parents else concept_name

    def _find_relationship_type(self, concept1: str, concept2: str) -> str:
        """
        Determine relationship type between two concepts.

        Args:
            concept1: First concept name
            concept2: Second concept name

        Returns:
            Relationship type string
        """
        c1 = self.find_concept(concept1)
        c2 = self.find_concept(concept2)

        if not c1 or not c2:
            return "related"

        from owlready2 import ThingClass

        # Check if c1 is subclass of c2
        c2_parents = [p for p in c1.is_a if isinstance(p, ThingClass)]
        if c2 in c2_parents:
            return "is_a"

        # Check if c1 is superclass of c2
        c1_parents = [p for p in c2.is_a if isinstance(p, ThingClass)]
        if c1 in c1_parents:
            return "is_a"

        # Default to related
        return "related"

    def __repr__(self) -> str:
        """String representation."""
        if self.ontology:
            stats = self.get_statistics()
            return (
                f"DomainOntologyManager({self.ontology_name}, "
                f"{stats['total_classes']} classes)"
            )
        return f"DomainOntologyManager({self.ontology_name}, not loaded)"
