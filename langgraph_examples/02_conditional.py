from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal


class JobState(TypedDict):
    score: int
    result: str


def check_score(state: JobState):
    print("Checking score...")
    return {}


def decide_next(state: JobState) -> Literal["shortlist", "improve"]:
    if state["score"] >= 70:
        return "shortlist"
    else:
        return "improve"


def shortlist_node(state: JobState):
    print("Candidate shortlisted")
    return {
        "result": "Candidate is shortlisted for the next interview."
    }


def improve_node(state: JobState):
    print("Candidate needs improvement")
    return {
        "result": "Candidate needs to improve skills before applying again."
    }


workflow = StateGraph(JobState)

workflow.add_node("check", check_score)
workflow.add_node("shortlist", shortlist_node)
workflow.add_node("improve", improve_node)

workflow.add_conditional_edges(
    "check",
    decide_next,
    {
        "shortlist": "shortlist",
        "improve": "improve"
    }
)

workflow.add_edge("shortlist", END)
workflow.add_edge("improve", END)

workflow.add_edge(START, "check")

app = workflow.compile()


# Get score from user
score = int(input("Enter candidate score: "))


# Run the graph
result = app.invoke({
    "score": score,
    "result": ""
})


print("\nFINAL RESULT:")
print(result)