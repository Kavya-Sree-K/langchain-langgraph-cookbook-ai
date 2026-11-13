from dotenv import load_dotenv
load_dotenv()

# TYPE: RAG as a pure LCEL chain (retriever | prompt | LLM)
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

PDF = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.pdf"

docs = PyPDFLoader(str(PDF)).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

vector_db = FAISS.from_documents(chunks, GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"))
retriever = vector_db.as_retriever(search_kwargs={"k": 3})


def format_docs(documents):
    return "\n\n".join(d.page_content for d in documents)


prompt = ChatPromptTemplate.from_template(
    "Answer using ONLY the context. If it is not there, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke(input("Enter your question: ")))
