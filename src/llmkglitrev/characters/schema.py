"""Character schema definitions."""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import datetime


class ResearchCharacter(BaseModel):
    """
    Reusable research character configuration.

    A character represents a domain expert with specific knowledge, methods,
    and personality traits. Characters can be saved and reused across multiple
    research sessions.
    """

    # Identity
    character_id: str = Field(
        description="Unique identifier (e.g., 'ml_expert_critical')"
    )
    name: str = Field(
        description="Display name (e.g., 'Dr. Machine Learning')"
    )

    # Domain Knowledge
    domain: str = Field(
        description="Primary research domain (e.g., 'Machine Learning')"
    )
    sub_domains: List[str] = Field(
        default_factory=list,
        description="Specific sub-domains of expertise",
        examples=[["deep learning", "computer vision", "NLP"]]
    )

    # Field-Specific Knowledge
    typical_venues: List[str] = Field(
        default_factory=list,
        description="Common publication venues for this domain",
        examples=[["NeurIPS", "ICML", "ICLR", "CVPR"]]
    )

    typical_methods: List[str] = Field(
        default_factory=list,
        description="Standard methodological approaches in this domain",
        examples=[["supervised learning", "transfer learning", "fine-tuning"]]
    )

    typical_datasets: List[str] = Field(
        default_factory=list,
        description="Common datasets used in this domain",
        examples=[["ImageNet", "COCO", "GLUE"]]
    )

    theoretical_foundations: List[str] = Field(
        default_factory=list,
        description="Key theories and frameworks",
        examples=[["statistical learning theory", "information theory"]]
    )

    # Character Traits
    stance: Literal["critical", "constructive", "neutral", "pragmatic"] = Field(
        default="neutral",
        description="Overall attitude toward research proposals"
    )

    communication_style: str = Field(
        default="academic",
        description="How the character communicates"
    )

    focus_areas: List[str] = Field(
        default_factory=list,
        description="What aspects the character pays attention to",
        examples=[["methodology", "novelty", "feasibility"]]
    )

    # Dialogue Configuration
    question_types_to_ask: List[str] = Field(
        default_factory=list,
        description="Types of questions this character should focus on",
        examples=[["methodological_justification", "assumption_challenge"]]
    )

    # Metadata
    created_by: str = Field(
        default="system",
        description="User who created this character"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When the character was created"
    )
    last_modified: datetime = Field(
        default_factory=datetime.now,
        description="When the character was last modified"
    )
    version: str = Field(
        default="1.0",
        description="Character version"
    )
    description: Optional[str] = Field(
        default=None,
        description="Free-text description of character's role"
    )

    # System Prompt Template
    system_prompt_template: str = Field(
        default="",
        description="Template for generating system prompts (with placeholders)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "character_id": "ml_expert_critical",
                    "name": "Dr. Machine Learning",
                    "domain": "Machine Learning",
                    "sub_domains": ["deep learning", "computer vision"],
                    "typical_venues": ["NeurIPS", "ICML", "CVPR"],
                    "stance": "critical",
                    "communication_style": "academic",
                    "focus_areas": ["methodology", "novelty"]
                }
            ]
        }
    }
