from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

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


class Source(BaseModel):
    """Schema of a source used by Agent"""

    source: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for Agent rsponse with answer and sources"""

    answer: str = Field(description="The agent answer key")
    sources: List[Source] = Field(
        default_factory=list,
        description="list of sources used by agent to generate answe key",
    )


def main():
    print("Hello from langchain-course!")

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
    agent = create_agent(
        model=llm, tools=[TavilySearch()], response_format=AgentResponse
    )
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Find 3 job openings for Gen AI developer in Linkedin"
                )
            ]
        }
    )
    print(response)


if __name__ == "__main__":
    main()
