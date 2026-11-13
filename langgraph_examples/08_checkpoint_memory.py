from dotenv import load_dotenv
load_dotenv()

# TYPE: Persistent memory with a checkpointer (thread_id = one conversation)
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def chatbot(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


workflow = StateGraph(MessagesState)
workflow.add_node("chatbot", chatbot)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

app = workflow.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "user-1"}}

while True:
    q = input("\nYou (type exit to stop): ")
    if q.lower() == "exit":
        break
    result = app.invoke({"messages": [{"role": "user", "content": q}]}, config)
    print("AI:", result["messages"][-1].content)
