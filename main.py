from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

load_dotenv()


# @tool
# def search(query: str) -> str:
#     """
#     Tool that search weather over internet
#     Args:
#         query: The query to search for
#     Return:
#         The search result
#     """
#     print(f"searching for {query}")
#     tavily = TavilySearch()
#     return tavily.invoke({"query": query,"follow_up_questions":1})


def main():
    print("Hello from langchain-course!")

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
    agent = create_agent(model=llm, tools=[TavilySearch()])
    response = agent.invoke(
        {"messages": [HumanMessage(content="What is the weather in Tokyo?")]}
    )
    print(response)


if __name__ == "__main__":
    main()
