from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI




def main():
    print("Hello from langchain-course!")


if __name__ == "__main__":
    main()
