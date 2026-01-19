"""
Test script for loading and exploring ontologies using Owlready2.

This script demonstrates:
1. Loading ontologies from web URLs
2. Exploring classes and properties
3. Querying relationships
4. Extracting domain knowledge
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from owlready2 import *

def test_load_edam_ontology():
    """
    Test loading EDAM ontology (data analysis and management).
    EDAM includes bioimaging and machine learning concepts.
    """
    print("="*80)
    print("TEST 1: Loading EDAM Ontology from Web")
    print("="*80)

    try:
        # Load EDAM from official URL
        edam_url = "http://edamontology.org/EDAM.owl"
        print(f"\n📥 Loading EDAM ontology from: {edam_url}")
        print("   (This may take a minute...)")

        edam = get_ontology(edam_url).load()

        print(f"✅ Successfully loaded EDAM ontology!")
        print(f"   IRI: {edam.base_iri}")

        # Get all classes
        classes = list(edam.classes())
        print(f"\n📊 Statistics:")
        print(f"   Total classes: {len(classes)}")

        # Look for machine learning related classes
        ml_classes = [c for c in classes if 'machine' in str(c.name).lower() or 'learning' in str(c.name).lower()]

        if ml_classes:
            print(f"\n🤖 Machine Learning Related Classes ({len(ml_classes)}):")
            for cls in ml_classes[:10]:  # Show first 10
                print(f"   - {cls.name}")
                if cls.comment:
                    print(f"     Comment: {cls.comment[0][:100]}...")

        # Look for imaging related classes
        imaging_classes = [c for c in classes if 'imag' in str(c.name).lower()]

        if imaging_classes:
            print(f"\n📸 Imaging Related Classes ({len(imaging_classes)}):")
            for cls in imaging_classes[:5]:
                print(f"   - {cls.name}")

        return edam

    except Exception as e:
        print(f"❌ Error loading EDAM: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_load_ml_schema():
    """
    Test loading ML-Schema from W3C GitHub.
    """
    print("\n" + "="*80)
    print("TEST 2: Loading ML-Schema Ontology from W3C")
    print("="*80)

    try:
        # Load ML-Schema from GitHub (Turtle format)
        ml_schema_url = "https://raw.githubusercontent.com/ML-Schema/core/master/MLSchema.ttl"
        print(f"\n📥 Loading ML-Schema from: {ml_schema_url}")

        mlschema = get_ontology(ml_schema_url).load()

        print(f"✅ Successfully loaded ML-Schema!")
        print(f"   IRI: {mlschema.base_iri}")

        # Get all classes
        classes = list(mlschema.classes())
        print(f"\n📊 Statistics:")
        print(f"   Total classes: {len(classes)}")

        # Display some key classes
        print(f"\n🔍 Sample Classes:")
        for cls in classes[:15]:
            print(f"   - {cls.name}")
            if hasattr(cls, 'comment') and cls.comment:
                comment = cls.comment[0] if isinstance(cls.comment, list) else cls.comment
                print(f"     → {comment[:80]}...")

        # Get properties
        properties = list(mlschema.object_properties()) + list(mlschema.data_properties())
        print(f"\n🔗 Properties: {len(properties)}")
        for prop in properties[:10]:
            print(f"   - {prop.name}")

        return mlschema

    except Exception as e:
        print(f"❌ Error loading ML-Schema: {e}")
        import traceback
        traceback.print_exc()
        return None


def explore_ontology_relationships(onto):
    """
    Explore relationships and hierarchy in an ontology.
    """
    if not onto:
        print("⚠️  No ontology to explore")
        return

    print("\n" + "="*80)
    print("TEST 3: Exploring Ontology Relationships")
    print("="*80)

    # Get all classes
    classes = list(onto.classes())

    if not classes:
        print("⚠️  No classes found in ontology")
        return

    print(f"\n📚 Exploring class hierarchy...")

    # Find root classes (classes with no superclass except owl:Thing)
    root_classes = []
    for cls in classes[:20]:  # Sample first 20
        supers = [s for s in cls.is_a if isinstance(s, ThingClass)]
        if not supers or all(s.name == 'Thing' for s in supers):
            root_classes.append(cls)

    print(f"\n🌳 Root Classes ({len(root_classes)}):")
    for cls in root_classes[:5]:
        print(f"\n   {cls.name}")

        # Get subclasses
        subclasses = list(cls.subclasses())
        if subclasses:
            print(f"   └─ Subclasses ({len(subclasses)}):")
            for sub in subclasses[:3]:
                print(f"      • {sub.name}")

    # Look for specific ML-related concepts
    print(f"\n🔍 Searching for specific concepts...")

    search_terms = ['algorithm', 'model', 'training', 'dataset', 'evaluation',
                   'neural', 'network', 'supervised', 'transfer', 'learning']

    for term in search_terms:
        matching = [c for c in classes if term.lower() in str(c.name).lower()]
        if matching:
            print(f"\n   '{term}' → Found {len(matching)} classes:")
            for cls in matching[:3]:
                print(f"      • {cls.name}")


def query_with_sparql(onto):
    """
    Demonstrate SPARQL queries on the ontology.
    """
    print("\n" + "="*80)
    print("TEST 4: SPARQL Queries")
    print("="*80)

    try:
        # Get RDFlib graph for SPARQL
        graph = default_world.as_rdflib_graph()

        print(f"\n📊 Graph has {len(graph)} triples")

        # Simple SPARQL query to find all classes
        query = """
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?class ?label WHERE {
            ?class a owl:Class .
            OPTIONAL { ?class rdfs:label ?label }
        }
        LIMIT 10
        """

        print("\n🔍 SPARQL Query Results (first 10 classes):")
        results = graph.query(query)

        for i, row in enumerate(results, 1):
            class_uri = row['class']
            label = row['label'] if row['label'] else 'N/A'
            print(f"   {i}. {label}")
            print(f"      URI: {class_uri}")

    except Exception as e:
        print(f"⚠️  SPARQL query failed: {e}")


def extract_domain_knowledge(onto, search_term="transfer learning"):
    """
    Extract knowledge about a specific domain concept.
    """
    print("\n" + "="*80)
    print(f"TEST 5: Extracting Knowledge about '{search_term}'")
    print("="*80)

    classes = list(onto.classes())

    # Search for classes related to the term
    search_words = search_term.lower().split()
    matching_classes = []

    for cls in classes:
        cls_name_lower = str(cls.name).lower()
        if any(word in cls_name_lower for word in search_words):
            matching_classes.append(cls)

    if not matching_classes:
        print(f"\n⚠️  No classes found matching '{search_term}'")
        return

    print(f"\n✅ Found {len(matching_classes)} related classes:")

    for cls in matching_classes:
        print(f"\n📚 Class: {cls.name}")

        # Get comments/definitions
        if hasattr(cls, 'comment') and cls.comment:
            comment = cls.comment[0] if isinstance(cls.comment, list) else cls.comment
            print(f"   Definition: {comment}")

        # Get superclasses
        supers = [s for s in cls.is_a if isinstance(s, ThingClass) and s.name != 'Thing']
        if supers:
            print(f"   Superclasses:")
            for sup in supers:
                print(f"      • {sup.name}")

        # Get subclasses
        subs = list(cls.subclasses())
        if subs:
            print(f"   Subclasses:")
            for sub in subs[:5]:
                print(f"      • {sub.name}")

        # Get related properties
        try:
            props = list(cls.get_class_properties())
            if props:
                print(f"   Properties:")
                for prop in props[:5]:
                    print(f"      • {prop.name}")
        except:
            pass


def main():
    """Main test function."""
    print("\n" + "="*80)
    print("🧪 OWLREADY2 ONTOLOGY LOADING TESTS")
    print("="*80)
    print("\nThis script tests loading ontologies from web sources.")
    print("We'll try EDAM and ML-Schema ontologies.\n")

    # Test 1: Try loading EDAM (smaller, more reliable)
    edam = test_load_edam_ontology()

    if edam:
        explore_ontology_relationships(edam)
        query_with_sparql(edam)
        extract_domain_knowledge(edam, "machine learning")

    # Test 2: Try loading ML-Schema
    print("\n\n")
    mlschema = test_load_ml_schema()

    if mlschema:
        explore_ontology_relationships(mlschema)
        extract_domain_knowledge(mlschema, "supervised learning")

    # Summary
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)

    if edam:
        print("✅ EDAM ontology loaded successfully")
    else:
        print("❌ EDAM ontology failed to load")

    if mlschema:
        print("✅ ML-Schema ontology loaded successfully")
    else:
        print("❌ ML-Schema ontology failed to load")

    if edam or mlschema:
        print("\n✅ At least one ontology loaded successfully!")
        print("   You can now use these ontologies to enhance research agents.")
    else:
        print("\n⚠️  No ontologies loaded. Check internet connection or URLs.")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
