from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from langchain_community.tools import DuckDuckGoSearchRun


# 1. Define state

class ResearchState(TypedDict):
    topic: str
    number_of_results: int
    plan: str
    search_results: str
    analysis: str
    final_answer: str


# 2. Create search tool

search_tool = DuckDuckGoSearchRun()


# 3. Plan node

def plan_node(state: ResearchState):

    topic = state["topic"]
    number = state["number_of_results"]

    plan = f"Search for {number} useful results about {topic} and analyse them."

    print("\nPLAN:")
    print(plan)

    return {
        "plan": plan
    }


# 4. Search node

def search_node(state: ResearchState):

    topic = state["topic"]

    print("\nSEARCHING WEB...")

    results = search_tool.invoke(topic)

    return {
        "search_results": results
    }


# 5. Analyse node

def analyse_node(state: ResearchState):

    results = state["search_results"]

    print("\nANALYSING RESULTS...")

    analysis = results[:3000]

    return {
        "analysis": analysis
    }


# 6. Final output node

def final_node(state: ResearchState):

    topic = state["topic"]
    analysis = state["analysis"]

    answer = f"""
Research topic: {topic}

Search findings:

{analysis}
"""

    return {
        "final_answer": answer
    }


# 7. Build graph

workflow = StateGraph(ResearchState)

workflow.add_node("plan", plan_node)
workflow.add_node("search", search_node)
workflow.add_node("analyse", analyse_node)
workflow.add_node("final", final_node)


# 8. Connect nodes

workflow.add_edge(START, "plan")
workflow.add_edge("plan", "search")
workflow.add_edge("search", "analyse")
workflow.add_edge("analyse", "final")
workflow.add_edge("final", END)


# 9. Compile graph

app = workflow.compile()


# 10. User input

topic = input("Enter research topic: ")
number = int(input("How many results do you want? "))


# 11. Run graph

result = app.invoke({
    "topic": topic,
    "number_of_results": number,
    "plan": "",
    "search_results": "",
    "analysis": "",
    "final_answer": ""
})


# 12. Print final answer

print("\nFINAL ANSWER:")
print(result["final_answer"])