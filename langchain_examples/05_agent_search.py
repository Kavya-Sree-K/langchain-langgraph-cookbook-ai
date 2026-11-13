from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

agent = create_agent(
    model=llm,
    tools=[search]
)

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What are the latest developments in AI?"
        }
    ]
})

print(response["messages"][-1].content)