"""Industry Partner Dialogue System.

Similar to Socratic Dialogue, but for industry partners asking about:
- Project fit
- Alignment/reusability
- Cost/resources

Uses uploaded project documents (PDF/DOCX) for context-aware answers.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model


class IndustryPartner(BaseModel):
    """Industry partner configuration."""
    partner_id: str = Field(description="Unique partner ID")
    name: str = Field(description="Partner name (e.g., 'Project Manager')")
    focus_area: str = Field(description="What they focus on")
    conversation_theme: str = Field(description="Main theme: project_fit, alignment, or cost")


class IndustryDialogue:
    """Chat interface for industry partner questions about project alignment."""

    # Define three industry partners
    PARTNERS = [
        IndustryPartner(
            partner_id="project_manager",
            name="Project Manager",
            focus_area="Project alignment, scope, and fit",
            conversation_theme="project_fit"
        ),
        IndustryPartner(
            partner_id="technical_lead",
            name="Technical Lead",
            focus_area="Technical alignment, reusability, and integration",
            conversation_theme="alignment"
        ),
        IndustryPartner(
            partner_id="stakeholder",
            name="Business Stakeholder",
            focus_area="Cost, resources, and ROI",
            conversation_theme="cost"
        )
    ]

    def __init__(self, session_id: str):
        """Initialize industry dialogue for a session.

        Args:
            session_id: The session ID containing proposal and documents
        """
        self.session_id = session_id

        # Load final proposal
        from pathlib import Path
        proposal_path = Path(f"sessions/{session_id}/final_proposal.md")
        if proposal_path.exists():
            self.final_proposal = proposal_path.read_text()
        else:
            self.final_proposal = "No final proposal found."

        # Load project document RAG if available
        self.project_rag = None
        try:
            from llmkglitrev.utils.document_rag import ProjectDocumentRAG
            self.project_rag = ProjectDocumentRAG(session_id)
            if self.project_rag.load_index():
                print(f"✓ Loaded project document index")
            else:
                print(f"ℹ️  No project documents indexed yet")
                self.project_rag = None
        except Exception as e:
            print(f"⚠️  Could not load project documents: {e}")
            self.project_rag = None

        # Initialize LLM
        self.llm = init_chat_model(model="deepseek:deepseek-chat")

    def get_available_partners(self) -> List[Dict[str, str]]:
        """Get list of available industry partners for dialogue."""
        return [
            {
                "partner_id": p.partner_id,
                "name": p.name,
                "focus_area": p.focus_area,
                "conversation_theme": p.conversation_theme
            }
            for p in self.PARTNERS
        ]

    async def generate_questions(
        self,
        partner_id: str,
        num_questions: int = 5
    ) -> List[str]:
        """Generate questions from an industry partner's perspective.

        The partner reads:
        - Research proposal
        - Project documents (if uploaded)

        And asks questions about project alignment, implementation, or cost.

        Args:
            partner_id: Which partner to use
            num_questions: How many questions to generate

        Returns:
            List of questions
        """
        # Find partner
        partner = next((p for p in self.PARTNERS if p.partner_id == partner_id), None)
        if not partner:
            raise ValueError(f"Unknown partner: {partner_id}")

        # Get project context if available
        project_context = ""
        project_summary = ""
        if self.project_rag:
            # FIX 1: Retrieve project docs using broad queries to get key project details
            # NOT based on research proposal - get actual project information
            broad_queries = [
                f"{partner.focus_area} project overview system architecture",  # Get system details
                "project objectives goals requirements constraints",  # Get business context
                "current system technologies infrastructure"  # Get technical details
            ]

            # Retrieve chunks from multiple queries to ensure we capture project specifics
            all_contexts = []
            for query in broad_queries:
                contexts = self.project_rag.retrieve_context(query, k=3)
                all_contexts.extend(contexts)

            # Deduplicate based on content
            seen_content = set()
            unique_contexts = []
            for ctx in all_contexts:
                content_hash = hash(ctx['content'][:100])  # Hash first 100 chars
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_contexts.append(ctx)

            # Format top 7 unique chunks
            formatted = ["<Project Documentation Context>"]
            for i, ctx in enumerate(unique_contexts[:7], 1):
                source = ctx['metadata'].get('source', 'Unknown')
                formatted.append(f"\n[Document {i}: {source}]")
                formatted.append(ctx['content'])
            formatted.append("\n</Project Documentation Context>")
            project_context = "\n".join(formatted)

            # Create a summary of key project details for emphasis
            doc_summary = self.project_rag.get_document_summary()
            project_summary = f"\nProject has {doc_summary['num_documents']} document(s): {', '.join([d['name'] for d in doc_summary['documents']])}\n"

        # Calculate depth distribution (same as Socratic dialogue)
        num_general = max(1, num_questions // 3)
        num_intermediate = max(1, (num_questions - num_general) // 2)
        num_deep = num_questions - num_general - num_intermediate

        # Theme-specific focus based on partner
        theme_guidance = self._get_theme_guidance(partner.conversation_theme)

        # Prompt to generate questions
        # FIX 2 & 3: Put project context FIRST, add explicit instructions to use it
        prompt = f"""You are a {partner.name} reviewing a research proposal for adoption into an existing project.

===== OUR PROJECT DOCUMENTATION =====
{project_summary}
{project_context}
=====================================

CRITICAL INSTRUCTIONS FOR QUESTION GENERATION:
1. READ the project documentation above CAREFULLY - it describes OUR current system
2. Your questions MUST reference SPECIFIC details from the project documents above
3. Use actual system names, technologies, constraints, and requirements mentioned in the docs
4. If docs mention "ticket system", your questions should mention "ticket system"
5. If docs mention specific technologies (e.g., "Python 3.8", "Jenkins"), reference them
6. Questions should assess: "How does THIS research proposal fit OUR specific project?"

===== RESEARCH PROPOSAL TO EVALUATE =====
{self.final_proposal}
=========================================

Your Role: {partner.focus_area}

Your task: Generate {num_questions} questions that assess if this research fits OUR SPECIFIC PROJECT (described in project docs above), focusing on: {partner.conversation_theme.replace('_', ' ').title()}

{theme_guidance}

IMPORTANT: Structure your questions with PROGRESSIVE DEPTH:

**First {num_general} question(s) - GENERAL LEVEL:**
- High-level fit with project goals and scope
- Overall alignment with project direction
- General feasibility concerns

**Next {num_intermediate} question(s) - INTERMEDIATE LEVEL:**
- Specific implementation considerations
- Integration with existing project infrastructure
- Resource and timeline implications

**Final {num_deep} question(s) - DEEP SPECIFIC LEVEL:**
- Detailed technical/operational challenges
- Specific cost items or resource requirements
- Risk mitigation strategies and contingencies

{'Use the project documentation context above to ask specific alignment questions.' if project_context else 'Note: No project documentation available - ask general alignment questions.'}

Format your response as a numbered list:
1. [General question about overall fit]
2. [General question continued if needed]
3. [Intermediate question about implementation]
4. [Intermediate question continued if needed]
5. [Deep question about specific challenges]
...

Generate exactly {num_questions} questions following this general → intermediate → deep progression."""

        # Get response from LLM
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        answer = response.content

        # Parse numbered list
        questions = []
        for line in answer.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove number prefix
                question = line
                for i, char in enumerate(line):
                    if char in ['.', ')', '-']:
                        question = line[i+1:].strip()
                        break
                if question:
                    questions.append(question)

        return questions[:num_questions]  # Ensure exact count

    async def answer_question(
        self,
        partner_id: str,
        user_question: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """Industry partner answers user's question with project context.

        Args:
            partner_id: Which partner is answering
            user_question: User's question
            conversation_history: Previous conversation messages

        Returns:
            Partner's answer
        """
        # Find partner
        partner = next((p for p in self.PARTNERS if p.partner_id == partner_id), None)
        if not partner:
            raise ValueError(f"Unknown partner: {partner_id}")

        # Get project context relevant to this question
        project_context = ""
        project_summary = ""
        if self.project_rag:
            # FIX: Retrieve using the user's question ONLY (not mixed with proposal)
            # This ensures we get project docs relevant to what user is asking
            project_context = self.project_rag.format_context_for_prompt(
                user_question,  # Just the question - more focused retrieval
                k=7  # Get more chunks for better context
            )

            # Add document summary
            doc_summary = self.project_rag.get_document_summary()
            project_summary = f"Project Documentation: {', '.join([d['name'] for d in doc_summary['documents']])}\n"

        # Build conversation history
        history_text = ""
        if conversation_history:
            history_text = "\n\nPrevious conversation:\n"
            for msg in conversation_history[-5:]:  # Last 5 messages
                if isinstance(msg, dict):
                    speaker = msg.get("speaker", "Unknown")
                    message = msg.get("message", "")
                else:
                    speaker = msg.speaker
                    message = msg.message
                history_text += f"{speaker}: {message}\n"

        # Theme-specific context
        theme_context = self._get_theme_context(partner.conversation_theme)

        # Answer prompt - put project context FIRST
        prompt = f"""You are a {partner.name} discussing a research proposal's fit with an existing project.

Your Role: {partner.focus_area}
Your Focus: {partner.conversation_theme.replace('_', ' ').title()}

===== OUR PROJECT DOCUMENTATION =====
{project_summary}
{project_context}
=====================================

{history_text}

{theme_context}

===== RESEARCH PROPOSAL =====
{self.final_proposal}
============================

User's Question: {user_question}

INSTRUCTIONS:
1. Read the project documentation carefully - it describes our current system
2. Reference SPECIFIC details from project docs in your answer (system names, technologies, constraints)
3. Provide practical, actionable answer from your {partner.conversation_theme.replace('_', ' ')} perspective
4. {'Use specific details from the project documentation to assess alignment.' if project_context else 'Provide general guidance since no project documentation is available.'}
5. Be constructive and specific. If you identify concerns, suggest concrete solutions or mitigations."""

        # Get response
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content

    def _get_theme_guidance(self, theme: str) -> str:
        """Get theme-specific guidance for question generation."""
        if theme == "project_fit":
            return """Focus on PROJECT FIT questions:
- Does this research align with our project's objectives and scope?
- How similar is this research to what our project aims to achieve?
- Would this research direction fit within our project timeline and deliverables?
- Are there any misalignments between research goals and project goals?"""

        elif theme == "alignment":
            return """Focus on ALIGNMENT/REUSABILITY questions:
- Can we reuse results from this research in our project?
- How would this research integrate with our existing project infrastructure?
- What would be required to adapt this research for our project needs?
- Are there dependencies or prerequisites for using these research results?"""

        elif theme == "cost":
            return """Focus on COST/RESOURCES questions:
- What would it cost to implement this research idea in our project context?
- What resources (people, time, budget) would be needed?
- What are the hidden costs or resource requirements?
- Is the investment justified by expected project benefits?"""

        return ""

    def _get_theme_context(self, theme: str) -> str:
        """Get theme-specific context for answering questions."""
        if theme == "project_fit":
            return """When answering, consider:
- Strategic alignment with project goals
- Scope fit and boundary matching
- Timeline compatibility
- Deliverable relevance"""

        elif theme == "alignment":
            return """When answering, consider:
- Technical integration points
- Reusability of research outputs
- Adaptation requirements
- Infrastructure compatibility"""

        elif theme == "cost":
            return """When answering, consider:
- Direct costs (people, tools, services)
- Indirect costs (training, integration, maintenance)
- Time investment and opportunity costs
- ROI and value proposition"""

        return ""
