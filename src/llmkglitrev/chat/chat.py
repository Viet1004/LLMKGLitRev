from langchain_aws import BedrockLLM
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek



class ChatModel:
    def __init__(self) -> None:
        pass

class BedrockChatModel(ChatModel):
    def __init__(self, model_id: str, region: str):
        super().__init__()
        self.model = BedrockLLM(model=model_id, region=region)

    def invoke(self, prompt: str) -> AIMessage:
        response = self.model(prompt)
        return response

class OpenAIChatModel(ChatModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = ChatOpenAI()
    def invoke(self, prompt: str) -> AIMessage:
        response = self.model.invoke([HumanMessage(prompt)])
        return response

class HuggingFaceChatModel(ChatModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = ChatHuggingFace()

    def invoke(self, prompt: str) -> AIMessage:
        response = self.model.invoke([HumanMessage(prompt)])
        return response

class AnthropicChatModel(ChatModel):
    def __init__(self, model: str = "claude-sonnet-4-5-20250929") -> None:
        super().__init__()
        self.model = ChatAnthropic(model_name=model, timeout=300, stop=["\nHuman:"])

    def invoke(self, prompt: str) -> AIMessage:
        response = self.model.invoke([HumanMessage(prompt)])
        return response

class DeepSeekChatModel(ChatModel):
    def __init__(self, model: str = "deepseek-chat") -> None:
        super().__init__()
        self.model = ChatDeepSeek(model=model)

    def invoke(self, prompt: str) -> AIMessage:
        response = self.model.invoke([HumanMessage(prompt)])
        return response
