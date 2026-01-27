"""
Predefined Ontologies for Research Character Creation

This module contains URLs and metadata for publicly available ontologies
that can be used to enhance research character domain knowledge.
"""

from typing import Dict, List
from pydantic import BaseModel


class OntologyOption(BaseModel):
    """Metadata for a predefined ontology option."""

    id: str
    name: str
    domain: str
    url: str
    format: str
    description: str
    source: str
    last_updated: str


# Predefined ontology options available for selection
PREDEFINED_ONTOLOGIES: Dict[str, OntologyOption] = {
    "cso": OntologyOption(
        id="cso",
        name="Computer Science Ontology (CSO)",
        domain="Computer Science",
        url="https://cso.kmi.open.ac.uk/download/cso.owl",
        format="OWL",
        description="Comprehensive taxonomy of ~15k computer science topics including AI, LLMs, GenAI, RAG, Prompt Engineering. Updated January 2025.",
        source="https://cso.kmi.open.ac.uk/downloads",
        last_updated="2025-01-17"
    ),
    "uco": OntologyOption(
        id="uco",
        name="Unified Cybersecurity Ontology (UCO)",
        domain="Cybersecurity",
        url="https://raw.githubusercontent.com/Ebiquity/Unified-Cybersecurity-Ontology/master/uco_1_5_rdf.owl",
        format="OWL/RDF",
        description="Unified ontology for cybersecurity with 13,486 triples and 420 classes. Integrates STIX, CVE, CCE, CVSS, CAPEC, CYBOX.",
        source="https://github.com/Ebiquity/Unified-Cybersecurity-Ontology",
        last_updated="2023-04"
    ),
    "oeo": OntologyOption(
        id="oeo",
        name="Open Energy Ontology (OEO)",
        domain="Energy & Smart Grid",
        url="https://raw.githubusercontent.com/OpenEnergyPlatform/ontology/production/src/ontology/oeo.owl",
        format="OWL",
        description="Domain ontology for energy system analysis, smart grids, and power systems. Actively maintained.",
        source="https://github.com/OpenEnergyPlatform/ontology",
        last_updated="2024"
    ),
    "nif": OntologyOption(
        id="nif",
        name="NLP Interchange Format (NIF)",
        domain="Natural Language Processing",
        url="https://raw.githubusercontent.com/NLP2RDF/ontologies/master/nif-core/nif-core.ttl",
        format="Turtle/RDF",
        description="RDF/OWL-based format for NLP interoperability. Core concepts: String, Context, annotations, linguistic features.",
        source="https://github.com/NLP2RDF/ontologies",
        last_updated="2024"
    ),
    "eira": OntologyOption(
        id="eira",
        name="EIRA - European Interoperability Reference Architecture",
        domain="Information Systems & Enterprise Architecture",
        url="https://joinup.ec.europa.eu/collection/european-interoperability-reference-architecture-eira/solution/eira-ontology",
        format="OWL/RDF",
        description="EIRA ontology for interoperable e-Government systems. Based on ArchiMate. Defines architectural building blocks (ABBs) for enterprise architecture.",
        source="https://joinup.ec.europa.eu/collection/european-interoperability-reference-architecture-eira/solution/eira-ontology",
        last_updated="2024-05"
    )
}


# Display options for dropdown (includes None and Other)
def get_ontology_dropdown_options() -> List[str]:
    """
    Get list of ontology options for dropdown display.

    Returns:
        List of display strings for dropdown including:
        - "None (No ontology)"
        - Predefined ontology names
        - "Other (Custom URL)"
    """
    options = ["None (No ontology)"]

    # Add predefined ontologies
    for onto_id, onto_info in PREDEFINED_ONTOLOGIES.items():
        options.append(f"{onto_info.name} - {onto_info.domain}")

    # Add custom option
    options.append("Other (Custom URL)")

    return options


def get_ontology_url_from_selection(selection: str) -> str:
    """
    Convert dropdown selection to ontology URL.

    Args:
        selection: Selected option from dropdown

    Returns:
        Ontology URL string (empty if None selected)
    """
    if selection == "None (No ontology)":
        return ""

    if selection == "Other (Custom URL)":
        return ""  # User will provide custom URL

    # Match selection to predefined ontology
    for onto_id, onto_info in PREDEFINED_ONTOLOGIES.items():
        if selection.startswith(onto_info.name):
            return onto_info.url

    return ""


def get_ontology_info_from_selection(selection: str) -> OntologyOption:
    """
    Get full ontology metadata from dropdown selection.

    Args:
        selection: Selected option from dropdown

    Returns:
        OntologyOption with full metadata, or None if not found
    """
    for onto_id, onto_info in PREDEFINED_ONTOLOGIES.items():
        if selection.startswith(onto_info.name):
            return onto_info

    return None


def get_ontology_selection_from_url(url: str) -> str:
    """
    Reverse lookup: Get dropdown selection text from URL.

    Args:
        url: Ontology URL

    Returns:
        Dropdown selection text
    """
    if not url or url.strip() == "":
        return "None (No ontology)"

    # Check if URL matches predefined ontology
    for onto_id, onto_info in PREDEFINED_ONTOLOGIES.items():
        if onto_info.url == url:
            return f"{onto_info.name} - {onto_info.domain}"

    # Custom URL
    return "Other (Custom URL)"
