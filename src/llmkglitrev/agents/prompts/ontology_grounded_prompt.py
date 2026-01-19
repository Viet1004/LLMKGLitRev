"""
Ontology-grounded prompts for research character answers.

These prompts ensure that character responses are grounded in
domain ontology concepts and relationships.
"""

ONTOLOGY_GROUNDED_ANSWER_PROMPT = """
You are answering the following question:

**Question:**
{question}

**Context from Previous Research:**
{research_context}

**Relevant Domain Ontology Concepts:**
{ontology_concepts}

**Concept Relationships:**
{concept_relationships}

**Instructions:**
1. Ground your answer in the provided ontology concepts
2. Explicitly reference ontology concepts when you use them (use bold: **concept**)
3. Explain relationships between concepts from the ontology
4. If introducing new terminology not in the ontology, explain why it's needed and how it relates to existing concepts
5. Provide a reasoning trace showing how you connected concepts step-by-step

**Format your response as follows:**

### Answer:
[Your comprehensive answer to the question, referencing ontology concepts in **bold**]

### Concepts Used:
- **Concept A** (from {ontology_name} ontology) - [brief explanation of how it's used]
- **Concept B** (from {ontology_name} ontology) - [brief explanation of how it's used]

### Concept Relationships:
- **Concept A** --[relationship_type]--> **Concept B**: [explain why this relationship is relevant to the answer]

### New Terminology (if any):
- **Term X**: [why this term is needed and how it relates to ontology concepts]

### Reasoning Trace:
1. Started with **Concept A** from question
2. Connected to **Concept B** via relationship R
3. ...

Remember: Always prioritize concepts from the {ontology_name} ontology when formulating your answer.
"""

ONTOLOGY_CONCEPTS_CONTEXT = """
Relevant {ontology_name} Ontology Concepts:

{concepts_list}

Concept Definitions:
{definitions}

Concept Relationships:
{relationships}
"""

ONTOLOGY_VALIDATION_PROMPT = """
You are validating a research answer against domain ontology.

**Answer to Validate:**
{answer_text}

**Domain Ontology:** {ontology_name}

**Available Concepts in Ontology:**
{available_concepts}

**Instructions:**
Identify:
1. Which ontology concepts are correctly used in the answer
2. Which terms are NOT in the ontology (potential new concepts or terminology issues)
3. Whether concept relationships are accurately represented
4. Missing ontology concepts that should have been referenced

Return your analysis as JSON:
{{
    "valid_concepts": ["concept1", "concept2", ...],
    "invalid_terms": [
        {{
            "term": "term1",
            "reason": "not in ontology",
            "suggested_mapping": "closest ontology concept if applicable"
        }}
    ],
    "missing_concepts": ["concept that should be mentioned", ...],
    "relationship_accuracy": "accurate/partial/inaccurate",
    "overall_grounding_score": 0.8
}}
"""

def format_concepts_context(
    ontology_name: str,
    concepts: list,
    definitions: dict,
    relationships: list
) -> str:
    """Format ontology concepts for prompt inclusion.

    Args:
        ontology_name: Name of the ontology
        concepts: List of concept names
        definitions: Dict mapping concept names to definitions
        relationships: List of relationship dicts

    Returns:
        Formatted string for prompt
    """
    # Format concepts list
    concepts_list = "\n".join(f"- **{concept}**" for concept in concepts)

    # Format definitions
    definitions_list = "\n".join(
        f"- **{concept}**: {definition}"
        for concept, definition in definitions.items()
    )

    # Format relationships with arrow notation for clarity
    if relationships:
        relationships_list = "\n".join(
            f"- **{rel['source']}** --[{rel['relationship']}]--> **{rel['target']}**"
            for rel in relationships
        )
    else:
        relationships_list = "(No explicit relationships found between these concepts)"

    return ONTOLOGY_CONCEPTS_CONTEXT.format(
        ontology_name=ontology_name,
        concepts_list=concepts_list,
        definitions=definitions_list,
        relationships=relationships_list
    )

def build_ontology_grounded_prompt(
    question: str,
    research_context: str,
    ontology_name: str,
    ontology_concepts: list,
    concept_definitions: dict,
    concept_relationships: list
) -> str:
    """Build complete ontology-grounded answer prompt.

    Args:
        question: The question to answer
        research_context: Context from previous research
        ontology_name: Name of the ontology being used
        ontology_concepts: List of relevant concept names
        concept_definitions: Dict of concept definitions
        concept_relationships: List of relationship dicts

    Returns:
        Complete formatted prompt
    """
    ontology_context = format_concepts_context(
        ontology_name=ontology_name,
        concepts=ontology_concepts,
        definitions=concept_definitions,
        relationships=concept_relationships
    )

    # Format relationships for main prompt with arrow notation
    relationships_formatted = "\n".join(
        f"- **{rel['source']}** --[{rel['relationship']}]--> **{rel['target']}**"
        for rel in concept_relationships
    ) if concept_relationships else "No explicit relationships identified between these concepts"

    return ONTOLOGY_GROUNDED_ANSWER_PROMPT.format(
        question=question,
        research_context=research_context,
        ontology_concepts=ontology_context,
        concept_relationships=relationships_formatted,
        ontology_name=ontology_name
    )
