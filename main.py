from dotenv import load_dotenv
import os
load_dotenv()

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from tavily import TavilyClient
# from langchain import HuggingFaceHub

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))



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
    return tavily_client.search(query = query, max_results=10)

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

tools = [search_web]
agent = create_agent(
    model=llm,
    tools=tools,
)


def main():
    print("Hello from react-search-agent-1!")
    result = agent.invoke({
        "messages": [
            HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")
        ]
    })

    print(result)


if __name__ == "__main__":
    main()
