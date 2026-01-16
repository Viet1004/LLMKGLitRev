"""
Visualization utilities for research reasoning and ontology concepts.

This module provides functions to create interactive visualizations of:
- Concept relationship graphs
- Reasoning trace timelines
- Multi-agent research dashboards
- Cross-domain concept mappings
"""

from typing import List, Dict, Optional, Set, Tuple
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path


def visualize_concept_relationships(
    concepts: List[str],
    relationships: List[Dict[str, str]],
    highlight_path: Optional[List[str]] = None,
    title: str = "Concept Relationship Graph"
) -> go.Figure:
    """Create interactive graph of concept relationships.

    Args:
        concepts: List of ontology concept names
        relationships: List of dicts with 'source', 'target', 'relationship'
        highlight_path: Optional list of concepts to highlight (reasoning path)
        title: Graph title

    Returns:
        Plotly figure with interactive concept graph
    """
    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    for concept in concepts:
        G.add_node(concept)

    # Add edges with relationship labels
    edge_labels = {}
    for rel in relationships:
        source = rel['source']
        target = rel['target']
        relationship = rel['relationship']

        G.add_edge(source, target)
        edge_labels[(source, target)] = relationship

    # Layout - use spring layout for nice spacing
    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except:
        # Fallback to circular layout if spring fails
        pos = nx.circular_layout(G)

    # Create edge traces
    edge_traces = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        # Get relationship label
        rel_label = edge_labels.get(edge, "")

        # Create arrow
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=2, color='#888'),
            hoverinfo='text',
            text=f"{edge[0]} --[{rel_label}]--> {edge[1]}",
            showlegend=False
        )
        edge_traces.append(edge_trace)

    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

        # Highlight path nodes
        if highlight_path and node in highlight_path:
            node_color.append('#FF6B6B')  # Red for highlighted
            node_size.append(30)
        else:
            node_color.append('#4ECDC4')  # Teal for normal
            node_size.append(20)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        textfont=dict(size=10, color='black'),
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white')
        ),
        hoverinfo='text',
        hovertext=node_text,
        showlegend=False
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=600
    )

    return fig


def visualize_reasoning_trace(
    reasoning_steps: List[str],
    concepts_per_step: Optional[List[List[str]]] = None,
    title: str = "Reasoning Trace Timeline"
) -> go.Figure:
    """Create timeline visualization of reasoning steps.

    Args:
        reasoning_steps: List of reasoning step descriptions
        concepts_per_step: Optional list of concepts used in each step
        title: Graph title

    Returns:
        Plotly figure with reasoning timeline
    """
    if not concepts_per_step:
        concepts_per_step = [[]] * len(reasoning_steps)

    # Create timeline data
    steps = list(range(1, len(reasoning_steps) + 1))
    concept_counts = [len(concepts) for concepts in concepts_per_step]

    fig = go.Figure()

    # Add trace for concept count per step
    fig.add_trace(go.Scatter(
        x=steps,
        y=concept_counts,
        mode='markers+lines+text',
        marker=dict(
            size=[15 + count * 3 for count in concept_counts],
            color=concept_counts,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="# Concepts")
        ),
        line=dict(color='#4ECDC4', width=2),
        text=[f"Step {s}" for s in steps],
        textposition="top center",
        hovertext=[
            f"<b>Step {i+1}</b><br>{step}<br><br><b>Concepts ({len(concepts)}):</b><br>" +
            "<br>".join(f"• {c}" for c in concepts[:5]) +
            (f"<br>...and {len(concepts)-5} more" if len(concepts) > 5 else "")
            for i, (step, concepts) in enumerate(zip(reasoning_steps, concepts_per_step))
        ],
        hoverinfo='text'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Reasoning Step",
        yaxis_title="Number of Concepts Referenced",
        hovermode='closest',
        plot_bgcolor='white',
        height=400,
        showlegend=False
    )

    return fig


def create_research_dashboard_data(
    artifacts: List[Dict]
) -> Dict[str, go.Figure]:
    """Create multiple visualizations for research dashboard.

    Args:
        artifacts: List of conversation artifacts

    Returns:
        Dict of figure names to Plotly figures
    """
    figures = {}

    # 1. Concepts by Character
    character_concepts = {}
    for art in artifacts:
        char_id = art.get('character_id', 'Unknown')
        concepts = art.get('grounded_concepts', [])
        character_concepts[char_id] = len(concepts)

    if character_concepts:
        fig_concepts = px.bar(
            x=list(character_concepts.keys()),
            y=list(character_concepts.values()),
            labels={'x': 'Character', 'y': '# Concepts'},
            title='Concepts Used by Each Character',
            color=list(character_concepts.values()),
            color_continuous_scale='Viridis'
        )
        fig_concepts.update_layout(showlegend=False, height=400)
        figures['concepts_by_character'] = fig_concepts

    # 2. Dialogue Note Priorities
    all_notes = []
    for art in artifacts:
        notes = art.get('dialogue_notes', [])
        for note in notes:
            all_notes.append({
                'character': art.get('character_id', 'Unknown'),
                'priority': note.get('priority', 5),
                'type': note.get('type', 'unknown')
            })

    if all_notes:
        import pandas as pd
        df_notes = pd.DataFrame(all_notes)

        fig_priorities = px.scatter(
            df_notes,
            x='character',
            y='priority',
            color='type',
            size=[10] * len(df_notes),
            title='Dialogue Note Priorities by Character',
            labels={'character': 'Character', 'priority': 'Priority (1-10)'}
        )
        fig_priorities.update_layout(height=400)
        figures['note_priorities'] = fig_priorities

    # 3. Concept Overlap Between Characters
    all_character_concepts = {}
    for art in artifacts:
        char_id = art.get('character_id', 'Unknown')
        concepts = set(art.get('grounded_concepts', []))
        all_character_concepts[char_id] = concepts

    if len(all_character_concepts) > 1:
        # Calculate overlap matrix
        characters = list(all_character_concepts.keys())
        overlap_matrix = []

        for char1 in characters:
            row = []
            for char2 in characters:
                if char1 == char2:
                    overlap = len(all_character_concepts[char1])
                else:
                    overlap = len(all_character_concepts[char1] & all_character_concepts[char2])
                row.append(overlap)
            overlap_matrix.append(row)

        fig_overlap = go.Figure(data=go.Heatmap(
            z=overlap_matrix,
            x=characters,
            y=characters,
            colorscale='Blues',
            text=overlap_matrix,
            texttemplate='%{text}',
            textfont={"size": 12},
            hovertemplate='%{y} ∩ %{x}: %{z} concepts<extra></extra>'
        ))

        fig_overlap.update_layout(
            title='Concept Overlap Between Characters',
            xaxis_title='Character',
            yaxis_title='Character',
            height=400
        )
        figures['concept_overlap'] = fig_overlap

    return figures


def visualize_cross_domain_mapping(
    source_concepts: List[str],
    target_concepts: List[str],
    mappings: List[Dict],
    source_domain: str,
    target_domain: str
) -> go.Figure:
    """Visualize concept mapping between domains.

    Args:
        source_concepts: Concepts from source domain
        target_concepts: Concepts from target domain
        mappings: List of mapping dicts with 'source', 'target', 'confidence'
        source_domain: Name of source domain
        target_domain: Name of target domain

    Returns:
        Plotly figure showing cross-domain mapping
    """
    # Create bipartite graph layout
    fig = go.Figure()

    # Source concepts on left (x=0)
    source_y = list(range(len(source_concepts)))
    source_x = [0] * len(source_concepts)

    # Target concepts on right (x=1)
    target_y = list(range(len(target_concepts)))
    target_x = [1] * len(target_concepts)

    # Draw mapping lines
    for mapping in mappings:
        source = mapping['source']
        target = mapping['target']
        confidence = mapping.get('confidence', 1.0)

        if source in source_concepts and target in target_concepts:
            source_idx = source_concepts.index(source)
            target_idx = target_concepts.index(target)

            # Color by confidence
            if confidence > 0.8:
                color = 'green'
            elif confidence > 0.6:
                color = 'orange'
            else:
                color = 'red'

            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[source_idx, target_idx],
                mode='lines',
                line=dict(color=color, width=2, dash='solid' if confidence > 0.7 else 'dash'),
                hovertext=f"{source} → {target}<br>Confidence: {confidence:.0%}",
                hoverinfo='text',
                showlegend=False
            ))

    # Add source nodes
    fig.add_trace(go.Scatter(
        x=source_x,
        y=source_y,
        mode='markers+text',
        marker=dict(size=15, color='#4ECDC4', line=dict(width=2, color='white')),
        text=source_concepts,
        textposition="middle left",
        textfont=dict(size=10),
        name=source_domain,
        hovertext=[f"<b>{c}</b><br>({source_domain})" for c in source_concepts],
        hoverinfo='text'
    ))

    # Add target nodes
    fig.add_trace(go.Scatter(
        x=target_x,
        y=target_y,
        mode='markers+text',
        marker=dict(size=15, color='#FF6B6B', line=dict(width=2, color='white')),
        text=target_concepts,
        textposition="middle right",
        textfont=dict(size=10),
        name=target_domain,
        hovertext=[f"<b>{c}</b><br>({target_domain})" for c in target_concepts],
        hoverinfo='text'
    ))

    fig.update_layout(
        title=f"Cross-Domain Concept Mapping: {source_domain} → {target_domain}",
        showlegend=True,
        hovermode='closest',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-0.3, 1.3]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        height=max(400, len(source_concepts) * 30),
        plot_bgcolor='white'
    )

    return fig


def export_visualization(
    fig: go.Figure,
    output_path: str,
    format: str = "html"
) -> str:
    """Export Plotly figure to file.

    Args:
        fig: Plotly figure
        output_path: Path to save file
        format: Output format ('html', 'png', 'svg', 'pdf')

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "html":
        fig.write_html(str(output_path))
    elif format in ["png", "svg", "pdf"]:
        # Requires kaleido package
        try:
            fig.write_image(str(output_path), format=format)
        except Exception as e:
            print(f"⚠️  Image export requires kaleido package: {e}")
            # Fall back to HTML
            html_path = output_path.with_suffix('.html')
            fig.write_html(str(html_path))
            return str(html_path)

    return str(output_path)


def create_mini_concept_graph(
    concepts: List[str],
    relationships: List[Dict[str, str]],
    max_nodes: int = 10
) -> str:
    """Create a small ASCII art concept graph for terminal display.

    Args:
        concepts: List of concept names
        relationships: List of relationship dicts
        max_nodes: Maximum nodes to display

    Returns:
        ASCII art string representation
    """
    # Limit to max nodes
    display_concepts = concepts[:max_nodes]

    # Build graph
    graph_lines = []
    graph_lines.append("Concept Graph:")
    graph_lines.append("=" * 60)

    # Group by relationships
    for rel in relationships[:max_nodes]:
        source = rel['source']
        target = rel['target']
        rel_type = rel['relationship']

        if source in display_concepts and target in display_concepts:
            # Create arrow representation
            arrow = f"{source} --[{rel_type}]--> {target}"
            graph_lines.append(f"  {arrow}")

    if len(concepts) > max_nodes:
        graph_lines.append(f"  ... and {len(concepts) - max_nodes} more concepts")

    graph_lines.append("=" * 60)

    return "\n".join(graph_lines)
