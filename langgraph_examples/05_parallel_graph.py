# TYPE: Parallel graph (fan-out / fan-in)
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class State(TypedDict):
    topic: str
    pros: str
    cons: str
    summary: str


def pros_node(state: State):
    print("PROS NODE RUNNING")
    return {"pros": f"{state['topic']} is fast and scalable."}


def cons_node(state: State):
    print("CONS NODE RUNNING")
    return {"cons": f"{state['topic']} can be costly and complex."}


def combine_node(state: State):
    print("COMBINE NODE RUNNING")
    return {"summary": f"PROS: {state['pros']} | CONS: {state['cons']}"}


workflow = StateGraph(State)
workflow.add_node("pros", pros_node)
workflow.add_node("cons", cons_node)
workflow.add_node("combine", combine_node)

# fan-out: two nodes start together
workflow.add_edge(START, "pros")
workflow.add_edge(START, "cons")

# fan-in: combine waits for BOTH
workflow.add_edge(["pros", "cons"], "combine")
workflow.add_edge("combine", END)

app = workflow.compile()

result = app.invoke({"topic": "Cloud computing", "pros": "", "cons": "", "summary": ""})
print("\nFINAL RESULT:")
print(result["summary"])
