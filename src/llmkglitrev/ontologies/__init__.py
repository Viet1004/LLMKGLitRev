"""Ontology management module for domain-specific knowledge representation."""

from .predefined_ontologies import (
    PREDEFINED_ONTOLOGIES,
    OntologyOption,
    get_ontology_dropdown_options,
    get_ontology_url_from_selection,
    get_ontology_info_from_selection,
    get_ontology_selection_from_url
)

__all__ = [
    "PREDEFINED_ONTOLOGIES",
    "OntologyOption",
    "get_ontology_dropdown_options",
    "get_ontology_url_from_selection",
    "get_ontology_info_from_selection",
    "get_ontology_selection_from_url"
]
