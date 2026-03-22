from pyexpat import model
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
load_dotenv()

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from langchain_tavily import TavilySearch

# from langchain import HuggingFaceHub

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Define a clear system prompt
SYSTEM_PROMPT = """You are a specialized Job Search Assistant.
Your goal is to find specific job postings and provide a structured summary.
When you have finished your research, you MUST provide your final answer 
using the 'AgentResponse' format, including the detailed answer and all 
source URLs used."""


class Source(BaseModel):
    """
     Schema for source used by the agent
    """
    url: str = Field(description="URL of the source")


class AgentResponse(BaseModel):
    """
    Schema for agent response with answer and sources
    """
    answer: str = Field(description="the agent answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer") 

@tool
def search_web(query: str) -> str:
    """
    Search the web for the query
    Args:
        query: The query to search for
    Returns:
        The search result
    """

    print(f"Searching the web for {query}")
    return tavily_client.search(query = query, max_results=3)

# hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# llm = HuggingFaceHub(
#     repo_id="Qwen/Qwen2.5-7B-Instruct",  # free model endpoint
#     model_kwargs={"temperature": 0, "max_new_tokens": 512},
#     huggingfacehub_api_token=hf_token
# )
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
)

tools = [TavilySearch(max_results=3)] # search_web is for python search instead TavilySearchResults()
agent = create_agent(
    model=llm,
    tools=tools,
    response_format= AgentResponse,
    system_prompt=SYSTEM_PROMPT
)


def main():
    print("Hello from react-search-agent-1!")
    result = agent.invoke({
        "messages": [
            HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")
        ]
    })

# 1. Print keys correctly as a list
    print(f"Available keys: {list(result.keys())}")

    # 2. Extract the AgentResponse object
    response = result.get("structured_response")
    
    if response:
        print("\n--- AGENT ANSWER ---")
        print(response.answer)
        print("\n--- SOURCES ---")
        for src in response.sources:
            print(f"- {src.url}")
    else:
        print("\nNo structured response found. The model might have returned text instead:")
        print(result["messages"][-1].content)

if __name__ == "__main__":
    main()
