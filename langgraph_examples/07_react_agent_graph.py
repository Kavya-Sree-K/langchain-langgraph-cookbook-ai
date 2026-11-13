from dotenv import load_dotenv
load_dotenv()

# TYPE: ReAct agent built by hand (LLM node + ToolNode + loop)
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


tools = [multiply, add]
llm_with_tools = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).bind_tools(tools)


def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


workflow = StateGraph(MessagesState)
workflow.add_node("assistant", assistant)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "assistant")
workflow.add_conditional_edges("assistant", tools_condition)  # -> "tools" or END
workflow.add_edge("tools", "assistant")                       # loop back

app = workflow.compile()

result = app.invoke({"messages": [{"role": "user", "content": "Add 5 and 7, then multiply the result by 6."}]})
print(result["messages"][-1].content)
