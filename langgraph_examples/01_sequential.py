from langgraph.graph import StateGraph, START, END
from typing import TypedDict


class MyState(TypedDict):
    query: str
    response: str


def plan_node(state: MyState):
    print("PLAN NODE RUNNING")
    return {
        "response": "Problem analyzed. Fixing plan created."
    }


def action_node(state: MyState):
    print("ACTION NODE RUNNING")
    return {
        "response": state["response"] + " Code checked and issue fixed."
    }


def summarize_node(state: MyState):
    print("SUMMARY NODE RUNNING")
    return {
        "response": state["response"] + " Final solution is ready."
    }


workflow = StateGraph(MyState)

workflow.add_node("plan", plan_node)
workflow.add_node("action", action_node)
workflow.add_node("summarize", summarize_node)

workflow.add_edge(START, "plan")
workflow.add_edge("plan", "action")
workflow.add_edge("action", "summarize")
workflow.add_edge("summarize", END)

app = workflow.compile()


result = app.invoke({
    "query": "Fix a Python login error",
    "response": ""
})

print("\nFINAL RESULT:")
print(result)