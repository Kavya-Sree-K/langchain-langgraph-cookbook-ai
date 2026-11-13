from dotenv import load_dotenv
load_dotenv()

# TYPE: Human-in-the-loop (interrupt, then resume with the human's decision)
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


class State(TypedDict):
    topic: str
    draft: str
    status: str


def draft_node(state: State):
    draft = llm.invoke(f"Write a 2-line announcement about {state['topic']}.").content
    return {"draft": draft}


def approval_node(state: State):
    decision = interrupt({"question": "Approve this draft? (yes/no)", "draft": state["draft"]})
    return {"status": "approved" if str(decision).lower() == "yes" else "rejected"}


def route(state: State):
    return "publish" if state["status"] == "approved" else "discard"


def publish_node(state: State):
    print("\nPUBLISHED:", state["draft"])
    return {}


def discard_node(state: State):
    print("\nDraft discarded.")
    return {}


workflow = StateGraph(State)
workflow.add_node("draft", draft_node)
workflow.add_node("approval", approval_node)
workflow.add_node("publish", publish_node)
workflow.add_node("discard", discard_node)

workflow.add_edge(START, "draft")
workflow.add_edge("draft", "approval")
workflow.add_conditional_edges("approval", route, {"publish": "publish", "discard": "discard"})
workflow.add_edge("publish", END)
workflow.add_edge("discard", END)

app = workflow.compile(checkpointer=MemorySaver())   # checkpointer is REQUIRED for interrupt
config = {"configurable": {"thread_id": "hitl-1"}}

result = app.invoke({"topic": "a new AI lab", "draft": "", "status": ""}, config)

info = result["__interrupt__"][0].value
print("\nDRAFT:\n", info["draft"])
answer = input(f"\n{info['question']} ")

app.invoke(Command(resume=answer), config)
