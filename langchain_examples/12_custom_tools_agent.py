from dotenv import load_dotenv
load_dotenv()

# TYPE: Agent with custom tools (@tool)
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def word_length(word: str) -> int:
    """Return the number of characters in a word."""
    return len(word)


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
agent = create_agent(model=llm, tools=[add, multiply, word_length])

response = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "What is (12 + 8) * 3, and how long is the word 'orchestration'?"
    }]
})

print(response["messages"][-1].content)
