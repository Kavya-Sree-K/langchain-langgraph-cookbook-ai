from dotenv import load_dotenv
load_dotenv()

# TYPE: Multi-agent supervisor (supervisor LLM routes to worker agents)
from typing import TypedDict, Literal
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


class Route(BaseModel):
    next: Literal["researcher", "writer", "FINISH"]


class State(TypedDict):
    task: str
    notes: str
    final: str
    next: str
    steps: int


supervisor_llm = llm.with_structured_output(Route)


def supervisor(state: State):
    decision = supervisor_llm.invoke(
        f"Task: {state['task']}\n"
        f"Research notes so far: {'yes' if state['notes'] else 'none'}\n"
        f"Final text so far: {'yes' if state['final'] else 'none'}\n"
        "Choose the next worker: 'researcher' (gather facts), "
        "'writer' (write the final answer from notes), or 'FINISH' (when final text exists)."
    )
    nxt = decision.next
    if state["steps"] >= 4:       # safety limit
        nxt = "FINISH"
    return {"next": nxt, "steps": state["steps"] + 1}


def researcher(state: State):
    print("RESEARCHER WORKING")
    notes = llm.invoke(f"List 5 key facts about: {state['task']}").content
    return {"notes": notes}


def writer(state: State):
    print("WRITER WORKING")
    final = llm.invoke(
        f"Using these notes, write a short paragraph.\n\nNotes:\n{state['notes']}\n\nTask: {state['task']}"
    ).content
    return {"final": final}


workflow = StateGraph(State)
workflow.add_node("supervisor", supervisor)
workflow.add_node("researcher", researcher)
workflow.add_node("writer", writer)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    lambda s: s["next"],
    {"researcher": "researcher", "writer": "writer", "FINISH": END},
)
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("writer", "supervisor")

app = workflow.compile()

result = app.invoke({"task": "Explain how a SOC uses SIEM and SOAR", "notes": "", "final": "", "next": "", "steps": 0})
print("\nFINAL:\n", result["final"])
