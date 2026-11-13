from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal


class PasswordState(TypedDict):
    password: str
    attempts: int
    message: str


MAX_ATTEMPTS = 3


def validate_node(state: PasswordState):
    print(f"Checking password - Attempt {state['attempts'] + 1}")

    password = state["password"]

    if len(password) >= 8:
        message = "Password is valid."
    else:
        message = "Password is too short."

    return {
        "message": message
    }


def improve_node(state: PasswordState):
    print("Password needs improvement.")

    new_password = input("Enter a new password: ")

    return {
        "password": new_password,
        "attempts": state["attempts"] + 1
    }


def check_node(state: PasswordState):
    return {}


def decide_next(state: PasswordState) -> Literal["improve", "end"]:

    if len(state["password"]) >= 8:
        return "end"

    if state["attempts"] < MAX_ATTEMPTS:
        return "improve"

    return "end"


workflow = StateGraph(PasswordState)

workflow.add_node("validate", validate_node)
workflow.add_node("check", check_node)
workflow.add_node("improve", improve_node)

workflow.add_edge(START, "validate")
workflow.add_edge("validate", "check")

workflow.add_conditional_edges(
    "check",
    decide_next,
    {
        "improve": "improve",
        "end": END
    }
)

workflow.add_edge("improve", "validate")

app = workflow.compile()


password = input("Enter your password: ")

result = app.invoke({
    "password": password,
    "attempts": 0,
    "message": ""
})

print("\nFINAL RESULT:")
print(result)