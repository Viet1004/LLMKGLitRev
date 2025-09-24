import fastapi
from llmkglitrev.chat import AnthropicChatModel, BedrockChatModel, HuggingFaceChatModel, OpenAIChatModel
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
import uvicorn
from fastapi import APIRouter, FastAPI
from langchain_aws import BedrockLLM
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful and just research assistant that is in charge of judging the quality of research ideas."
    ),
    ("human", "{query}"),
])

chat_model = ChatAnthropic(model_name="claude-3-5-sonnet-latest", timeout=300, stop=["\nHuman:"])

chain = prompt | chat_model
app = FastAPI(
    title="Chat_app",
    version="0.0.1",
    description="An API endpoint for chat models",
)

@app.get("/chat")
def chat(prompt: str):
    response = chain.invoke({"query": prompt})
    return {"response": response.content}

# api_router = APIRouter()

if __name__ == "__main__":
    uvicorn.run("llmkglitrev.api_endpoint.chat:app", host="0.0.0.0", port=8000, reload=True)

