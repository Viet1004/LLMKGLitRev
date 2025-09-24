import os

import gradio as gr
from dotenv import load_dotenv
from mcp import StdioServerParameters
from smolagents import CodeAgent, LiteLLMModel, ToolCollection

load_dotenv()

try:

    # Use ToolCollection.from_mcp with stdio configuration from agent.json
    server_parameters = StdioServerParameters(
        command="uv",
        args=[
            "--directory",
            "/Users/quocviet.nguyen/paper_idea/LLM_KG_LitRev/LLMKGLitRev/src/llmkglitrev/MCPServer/arxiv-mcp-server",
            "run",
            "arxiv-mcp-server",
            "--storage-path",
            "/Users/quocviet.nguyen/paper_idea/LLM_KG_LitRev/LLMKGLitRev/data/pdf",
        ],
        env=dict(os.environ),
    )

    with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
        # model = InferenceClientModel(token=os.getenv("HF_TOKEN"))
        model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-latest")
        agent = CodeAgent(
            tools=[*tool_collection.tools],
            model=model,
            additional_authorized_imports=["json", "ast", "urllib", "base64"],
            add_base_tools=True,
        )

        # agent = Agent(
        #     model=model,
        #     provider="nebius",
        #     servers=[
        #         {
        #             "command": "npx",
        #             "args": [
        #                 "mcp-remote",
        #                 "http://localhost:7860/gradio_api/mcp/sse"  # Your Gradio MCP server
        #             ]
        #         }
        #     ],
        #     # api_key=os.getenv("HF_TOKEN")
        # )

        demo = gr.ChatInterface(
            fn=lambda message, history: str(agent.run(message)),
            type="messages",
            examples=[""],
            title="Agent with MCP Tools",
            description="Extract and collect arxiv papers",
        )

        demo.launch()
finally:
    print("MCP session completed")
