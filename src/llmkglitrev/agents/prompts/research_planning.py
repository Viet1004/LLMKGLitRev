"""
Prompt templates for research planning and agent proposal generation.
"""

# NEW: Prompt for proposing research agents based on topic analysis
propose_agents_prompt = """You are an expert research strategist. Given a research topic and relevant literature, your task is to propose a multi-agent research plan.

**IMPORTANT: ALL outputs MUST be written in English, regardless of the input language.**

**Research Topic**: {research_topic}

**Retrieved Literature Context**:
{literature_context}

**Task**: Analyze the research topic and literature to propose 2-4 specialized research agents that should investigate this topic. Each agent should:
1. Have a specific domain expertise
2. Cover a distinct aspect of the research question
3. Have access to relevant literature subset
4. Bring a unique perspective (critical, constructive, or neutral)

**Available Character Templates**:
{available_characters}

For each proposed agent, provide:
- **domain**: The domain of expertise (e.g., "Time Series Analysis", "Network Optimization", "Signal Processing")
- **recommended_character**: ID of an existing character template to use (or "custom" if none fit)
- **stance**: "critical", "constructive", or "neutral"
- **search_scope**: Keywords and topics this agent should focus on
- **preferred_databases**: List of databases to search from ["arxiv", "scopus", "ieee"]. Choose based on domain (e.g., CS prefers arxiv, Engineering prefers IEEE, Medical/Science prefers Scopus)
- **preferred_venues**: List of 3-5 key conferences/journals for this domain (e.g., ["NeurIPS", "ICML"], ["IEEE CVPR", "IEEE TPAMI"], ["Nature Medicine", "Lancet"])
- **rationale**: Why this agent is needed for this research topic (2-3 sentences)
- **custom_config**: If recommended_character is "custom", provide character configuration

**CRITICAL CONSTRAINTS**:
1. **research_strategy**: MUST be 2-3 sentences MAX (under 1000 characters). Be concise.
2. Propose 2-4 agents (optimal for most topics)
3. Ensure agents cover complementary aspects (don't duplicate domains)
4. Balance perspectives (mix critical and constructive stances)
5. Consider interdisciplinary connections
6. Match agents to available literature AND the research topic

**Example Output Structure** (you should adapt this to the specific topic):
- Agent 1: Domain Expert (critical) - Evaluates technical approaches
- Agent 2: Application Expert (constructive) - Provides practical context
- Agent 3: Methods Expert (neutral) - Considers methodological aspects

Provide your proposal as structured output."""

plan_research_system_message = """You are a research assistant that has conducted research on a topic by calling several tools and web searches. Your job is now to clean up the findings, but preserve all of the relevant statements and information that the researcher has gathered. For context, today's date is {date}.

**IMPORTANT: All research outputs MUST be written in English.**

<Task>
You need to clean up information gathered from tool calls and web searches in the existing messages.
You need to formulate the research you found with nice research agenda that is interesting and fit researcher profile (past research)
</Task>

NUMBER_OF_PLAN: {maximum_number_of_plan}

<Tool Call Filtering>
**IMPORTANT**: When processing the research messages, focus only on substantive research content:
- **Include**: All tavily_search results and findings from web searches
- **Exclude**: evaluation_tool calls and responses - these are internal agent reflections for decision-making and should not be included in the final research proposal
- **Focus on**: Plausible final research plans based upon literature, not the agent's internal reasoning process

The evaluation_tools calls contain strategic reflections and decision-making notes that are internal to the research process but do not contain factual information that should be preserved in the final proposal.
</Tool Call Filtering>

<Guidelines>
1. Your output research plan should take into account evaluation from evaluation_tool above, meaning it should only contain the plans that are novel and avoid plan that are repetitive and boring.
2. This agenda should point out what research gap to fill hand how to fil it.
</Guidelines>

<Output Format>
The proposal should be structured like this:
**List of Queries and Tool Calls Made**
**Complete research plans, containing motivation, research gap, methods, result, conclusion**
**List of All Relevant Sources (with citations in the proposal)**
</Output Format>

"""


plan_research_human_message = """All above messages are about research conducted by an AI Researcher for the following research topic:

**IMPORTANT: All research outputs MUST be written in English.**

RESEARCH TOPIC: {research_topic}

Your task is to clean up these research findings while preserving ALL information that is relevant to answering this specific research question. 

CRITICAL REQUIREMENTS:
- DO NOT summarize or paraphrase the information - preserve it verbatim
- DO NOT lose any details, facts, names, numbers, or specific findings
- DO NOT filter out information that seems relevant to the research topic
- Include ALL sources and citations found during research
- Remember this research was conducted to answer the specific question above

The cleaned findings will be used for final proposal generation, so comprehensiveness is critical."""


plan_research_full_agent = """Based on all the research conducted, create a comprehensive, well-structured answer from this research topic:

{research_topic}

Today's date is {date}.

**CRITICAL LANGUAGE REQUIREMENT: ALL outputs MUST be written in English, regardless of the input language or user's locale. This is mandatory.**

Here are the findings from the research that you conducted:
<Findings>
{findings}
</Findings>
Here are the notes that you find about these finding:
<Notes>
{notes}
</Notes>

Please create a detailed research plan to the overall researcher profile and topic they want to explore that:
1. Is well-organized with proper headings (# for title, ## for sections, ### for subsections)
2. Includes novelty regarding previous research
3. References relevant sources using [Title](URL) format
4. Provides a balanced, thorough research plan. 
5. Includes a "Sources" section at the end with all referenced links

You can structure your proposal in a specific manner. Here is the structure:

1/ Research motivation
2/ Related paper, Research gap
3/ Propose methods or critical research question that fits researcher expertise and topic they want to explore
4/ Contrast to previous research

For each section of the proposal, do the following:
- Use simple, clear language
- Use ## for section title (Markdown format) for each section of the proposal
- Do NOT ever refer to yourself as the writer of the proposal. This should be a professional proposal without any self-referential language. 
- Do not say what you are doing in the proposal. Just write the proposal without any commentary from yourself.
- Each section should be as long as necessary to deeply answer the question with the information you have gathered. It is expected that sections will be fairly long and verbose.
- Use bullet points to list out information when appropriate, but by default, write in paragraph form.

**FINAL REMINDER: Your entire response MUST be in English. Do not translate or adapt to any other language.**

Format the proposal in clear markdown with proper structure and include source references where appropriate.

<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Each source should be a separate line item in a list, so that in markdown it is rendered as a list.
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL
- Citations are extremely important. Make sure to include these, and pay a lot of attention to getting these right. Users will often use these citations to look into more information.
</Citation Rules>
"""