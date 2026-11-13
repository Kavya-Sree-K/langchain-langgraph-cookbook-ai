from dotenv import load_dotenv
load_dotenv()

# TYPE: Agentic RAG (retrieve -> grade -> generate OR rewrite query and retry)
from pathlib import Path
from typing import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

PDF = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.pdf"
MAX_RETRIES = 2

docs = PyPDFLoader(str(PDF)).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
vector_db = FAISS.from_documents(chunks, GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"))

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


class Grade(BaseModel):
    relevant: bool = Field(description="True if the context can answer the question")


grader = llm.with_structured_output(Grade)


class State(TypedDict):
    question: str
    context: str
    relevant: bool
    retries: int
    answer: str


def retrieve(state: State):
    print("RETRIEVING for:", state["question"])
    results = vector_db.similarity_search(state["question"], k=3)
    return {"context": "\n\n".join(d.page_content for d in results)}


def grade(state: State):
    g = grader.invoke(
        f"Question: {state['question']}\n\nContext:\n{state['context']}\n\n"
        "Can the context answer the question?"
    )
    print("RELEVANT:", g.relevant)
    return {"relevant": g.relevant}


def rewrite(state: State):
    new_q = llm.invoke(
        f"Rewrite this question to be better for semantic search. Return only the question.\n\n{state['question']}"
    ).content.strip()
    return {"question": new_q, "retries": state["retries"] + 1}


def generate(state: State):
    answer = llm.invoke(
        f"Answer using ONLY this context. If missing, say you don't know.\n\n"
        f"Context:\n{state['context']}\n\nQuestion: {state['question']}"
    ).content
    return {"answer": answer}


def decide(state: State):
    if state["relevant"] or state["retries"] >= MAX_RETRIES:
        return "generate"
    return "rewrite"


workflow = StateGraph(State)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", decide, {"generate": "generate", "rewrite": "rewrite"})
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

app = workflow.compile()

q = input("Enter your question: ")
result = app.invoke({"question": q, "context": "", "relevant": False, "retries": 0, "answer": ""})
print("\nFINAL ANSWER:\n", result["answer"])
