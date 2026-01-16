"""
Cross-domain concept mapping utilities.

This module provides tools to map concepts between different domain ontologies,
enabling multidisciplinary research and industry-academia alignment.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import asyncio

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from llmkglitrev.ontology.ontology_manager import DomainOntologyManager
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ConceptMapping:
    """Represents a mapping between concepts from different domains."""
    source_concept: str
    source_domain: str
    target_concepts: List[str]
    target_domain: str
    mapping_confidence: float  # 0-1
    relationship_type: str  # "exact", "broader", "narrower", "related", "none"
    explanation: str
    suggested_alternatives: List[str] = None


@dataclass
class CrossDomainGap:
    """Represents a cross-domain knowledge gap."""
    gap_type: str  # "unmapped_concept", "weak_mapping", "missing_required_concept"
    source_concept: Optional[str] = None
    target_concept: Optional[str] = None
    target_concepts: Optional[List[str]] = None
    source_domain: Optional[str] = None
    target_domain: Optional[str] = None
    mapping_confidence: Optional[float] = None
    explanation: str = ""
    suggested_question: str = ""
    priority: int = 5  # 1-10


class ConceptMapper:
    """Map concepts between domain ontologies."""

    def __init__(
        self,
        ontology_managers: Dict[str, DomainOntologyManager],
        model_name: str = "deepseek:deepseek-chat"
    ):
        """Initialize concept mapper.

        Args:
            ontology_managers: Dict mapping domain names to ontology managers
            model_name: LLM model for assisted mapping
        """
        self.ontologies = ontology_managers

        # Try to initialize LLM, but allow mapper to work without it
        try:
            self.llm = init_chat_model(model=model_name)
        except Exception as e:
            print(f"⚠️  LLM initialization failed: {e}")
            print("   ConceptMapper will work with similarity-based mapping only")
            self.llm = None

        self.mapping_cache = {}

    def map_concept(
        self,
        concept: str,
        source_domain: str,
        target_domain: str
    ) -> ConceptMapping:
        """Map concept from source to target domain ontology.

        Args:
            concept: Concept name to map
            source_domain: Source domain name
            target_domain: Target domain name

        Returns:
            ConceptMapping with results
        """
        cache_key = f"{source_domain}:{concept}:{target_domain}"

        if cache_key in self.mapping_cache:
            return self.mapping_cache[cache_key]

        # Get ontology managers
        source_ontology = self.ontologies.get(source_domain)
        target_ontology = self.ontologies.get(target_domain)

        if not source_ontology or not target_ontology:
            return ConceptMapping(
                source_concept=concept,
                source_domain=source_domain,
                target_concepts=[],
                target_domain=target_domain,
                mapping_confidence=0.0,
                relationship_type="none",
                explanation="Ontology not available for one or both domains"
            )

        # Get concept definition from source
        source_def = source_ontology.get_concept_definition(concept)

        # Search for similar concepts in target domain
        target_candidates = target_ontology.search_similar_concepts(
            concept,
            similarity_threshold=0.5,
            max_results=5
        )

        if not target_candidates:
            # No candidates found - unmapped concept
            return ConceptMapping(
                source_concept=concept,
                source_domain=source_domain,
                target_concepts=[],
                target_domain=target_domain,
                mapping_confidence=0.0,
                relationship_type="none",
                explanation=f"No similar concepts found in {target_domain} ontology"
            )

        # Use simple similarity-based mapping
        # (In production, could use LLM for more sophisticated mapping)
        best_candidate = target_candidates[0]

        # Determine relationship type based on similarity
        similarity = best_candidate['similarity']

        if similarity > 0.9:
            relationship_type = "exact"
            confidence = similarity
        elif similarity > 0.7:
            relationship_type = "related"
            confidence = similarity * 0.9
        elif similarity > 0.5:
            relationship_type = "broader"  # or "narrower" - would need hierarchy check
            confidence = similarity * 0.8
        else:
            relationship_type = "weak"
            confidence = similarity * 0.6

        mapping = ConceptMapping(
            source_concept=concept,
            source_domain=source_domain,
            target_concepts=[best_candidate['name']],
            target_domain=target_domain,
            mapping_confidence=confidence,
            relationship_type=relationship_type,
            explanation=f"Mapped based on term similarity ({similarity:.0%})",
            suggested_alternatives=[c['name'] for c in target_candidates[1:3]]
        )

        self.mapping_cache[cache_key] = mapping
        return mapping

    async def map_concept_with_llm(
        self,
        concept: str,
        source_definition: str,
        source_domain: str,
        target_candidates: List[Dict],
        target_domain: str
    ) -> Dict:
        """Use LLM to determine best concept mapping.

        Args:
            concept: Source concept name
            source_definition: Definition from source ontology
            source_domain: Source domain name
            target_candidates: List of candidate concepts from target
            target_domain: Target domain name

        Returns:
            Mapping dict with target_concepts, confidence, relationship_type, explanation
        """
        # Check if LLM is available
        if not self.llm:
            # Fallback to similarity-based mapping
            if target_candidates:
                best = target_candidates[0]
                return {
                    "target_concepts": [best['name']],
                    "mapping_confidence": best['similarity'],
                    "relationship_type": "related",
                    "explanation": f"Similarity-based mapping (LLM unavailable, {best['similarity']:.0%} similarity)"
                }
            else:
                return {
                    "target_concepts": [],
                    "mapping_confidence": 0.0,
                    "relationship_type": "none",
                    "explanation": "No candidates found"
                }

        candidates_text = "\n".join(
            f"{i+1}. **{c['name']}**: {c['definition']} (similarity: {c['similarity']:.0%})"
            for i, c in enumerate(target_candidates)
        )

        prompt = f"""
You are mapping a concept from {source_domain} ontology to {target_domain} ontology.

**Source Concept:** {concept}
**Source Definition:** {source_definition}

**Candidate Concepts in {target_domain}:**
{candidates_text}

**Task:**
Determine the best mapping from the source concept to the target domain candidates.

**Relationship Types:**
- **exact**: The concepts mean the same thing (confidence: 0.9-1.0)
- **broader**: Source is a specific case of target (confidence: 0.7-0.9)
- **narrower**: Source is more general than target (confidence: 0.7-0.9)
- **related**: Concepts are related but not hierarchical (confidence: 0.5-0.8)
- **none**: No good mapping exists (confidence: < 0.5)

Return ONLY a JSON object (no markdown, no code blocks):
{{
    "target_concepts": ["concept_name"],
    "mapping_confidence": 0.85,
    "relationship_type": "exact",
    "explanation": "Brief explanation of why this mapping is appropriate"
}}
"""

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            import json
            import re

            # Clean response (remove markdown code blocks if present)
            content = response.content.strip()
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'\s*```$', '', content)

            result = json.loads(content)
            return result

        except Exception as e:
            print(f"LLM mapping failed: {e}, falling back to similarity-based")
            # Fallback to best candidate
            if target_candidates:
                best = target_candidates[0]
                return {
                    "target_concepts": [best['name']],
                    "mapping_confidence": best['similarity'],
                    "relationship_type": "related",
                    "explanation": f"Fallback similarity-based mapping ({best['similarity']:.0%})"
                }
            else:
                return {
                    "target_concepts": [],
                    "mapping_confidence": 0.0,
                    "relationship_type": "none",
                    "explanation": "No candidates found"
                }

    def map_multiple_concepts(
        self,
        concepts: List[str],
        source_domain: str,
        target_domain: str
    ) -> Dict[str, ConceptMapping]:
        """Map multiple concepts from source to target domain.

        Args:
            concepts: List of concept names
            source_domain: Source domain name
            target_domain: Target domain name

        Returns:
            Dict mapping source concepts to ConceptMapping objects
        """
        mappings = {}

        for concept in concepts:
            mapping = self.map_concept(concept, source_domain, target_domain)
            mappings[concept] = mapping

        return mappings

    def get_mapping_statistics(
        self,
        mappings: Dict[str, ConceptMapping]
    ) -> Dict:
        """Calculate statistics for a set of mappings.

        Args:
            mappings: Dict of concept mappings

        Returns:
            Dict with statistics
        """
        if not mappings:
            return {
                "total_concepts": 0,
                "mapped": 0,
                "unmapped": 0,
                "exact_matches": 0,
                "related_matches": 0,
                "weak_matches": 0,
                "average_confidence": 0.0
            }

        mapped = sum(1 for m in mappings.values() if m.target_concepts)
        unmapped = len(mappings) - mapped

        exact = sum(1 for m in mappings.values() if m.relationship_type == "exact")
        related = sum(1 for m in mappings.values() if m.relationship_type in ["related", "broader", "narrower"])
        weak = sum(1 for m in mappings.values() if m.relationship_type == "weak")

        confidences = [m.mapping_confidence for m in mappings.values() if m.target_concepts]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "total_concepts": len(mappings),
            "mapped": mapped,
            "unmapped": unmapped,
            "exact_matches": exact,
            "related_matches": related,
            "weak_matches": weak,
            "average_confidence": avg_confidence,
            "mapping_rate": mapped / len(mappings) if mappings else 0.0
        }


class CrossDomainGapIdentifier:
    """Identify knowledge gaps when viewing proposals through different domain lenses."""

    def __init__(self, concept_mapper: ConceptMapper):
        """Initialize gap identifier.

        Args:
            concept_mapper: ConceptMapper instance for mapping concepts
        """
        self.mapper = concept_mapper

    def identify_gaps(
        self,
        source_domain: str,
        target_domain: str,
        proposal_concepts: List[str],
        mapped_concepts: Dict[str, ConceptMapping]
    ) -> List[CrossDomainGap]:
        """Identify gaps when mapping between domains.

        Args:
            source_domain: Source domain name
            target_domain: Target domain name
            proposal_concepts: Concepts mentioned in proposal
            mapped_concepts: Mapping results from ConceptMapper

        Returns:
            List of cross-domain gaps
        """
        gaps = []

        # Type 1: Unmapped concepts
        unmapped = [
            c for c, m in mapped_concepts.items()
            if not m.target_concepts
        ]

        for concept in unmapped:
            gaps.append(CrossDomainGap(
                gap_type="unmapped_concept",
                source_concept=concept,
                source_domain=source_domain,
                target_domain=target_domain,
                explanation=f"'{concept}' from {source_domain} has no equivalent in {target_domain} ontology",
                suggested_question=f"How should '{concept}' be understood from a {target_domain} perspective?",
                priority=8
            ))

        # Type 2: Weak mappings (low confidence)
        weak_mappings = [
            (c, m) for c, m in mapped_concepts.items()
            if m.target_concepts and m.mapping_confidence < 0.7
        ]

        for concept, mapping in weak_mappings:
            gaps.append(CrossDomainGap(
                gap_type="weak_mapping",
                source_concept=concept,
                target_concepts=mapping.target_concepts,
                mapping_confidence=mapping.mapping_confidence,
                source_domain=source_domain,
                target_domain=target_domain,
                explanation=f"Uncertain mapping between domains (confidence: {mapping.mapping_confidence:.0%})",
                suggested_question=f"Is the {target_domain} interpretation of '{concept}' as '{mapping.target_concepts[0]}' appropriate?",
                priority=7
            ))

        # Type 3: Missing critical target domain concepts
        # (This would require domain-specific rules or analysis)
        # For now, we'll just identify concepts that should exist but don't

        return gaps

    def generate_cross_domain_questions(
        self,
        gaps: List[CrossDomainGap],
        max_questions: int = 10
    ) -> List[Dict]:
        """Generate questions to address cross-domain gaps.

        Args:
            gaps: List of identified gaps
            max_questions: Maximum questions to generate

        Returns:
            List of question dicts with priority, type, and question text
        """
        questions = []

        # Sort gaps by priority (highest first)
        sorted_gaps = sorted(gaps, key=lambda g: g.priority, reverse=True)

        for gap in sorted_gaps[:max_questions]:
            question = {
                "gap_type": gap.gap_type,
                "priority": gap.priority,
                "question": gap.suggested_question,
                "explanation": gap.explanation,
                "source_concept": gap.source_concept,
                "target_domain": gap.target_domain
            }
            questions.append(question)

        return questions
