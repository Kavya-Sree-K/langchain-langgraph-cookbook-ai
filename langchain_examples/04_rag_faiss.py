from dotenv import load_dotenv
import os
load_dotenv()

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field


api_key = os.getenv("GOOGLE_API_KEY")

loader = PyPDFLoader(str(Path(__file__).resolve().parent.parent / "data" / "knowledge_base.pdf"))
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Number of chunks:", len(chunks))

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)

vector_db = FAISS.from_documents(
    chunks,
    embeddings
)

print("Vector database created")

question = input("Enter your question: ")

results = vector_db.similarity_search_with_score(
    question,
    k=3
)

print("\nTop 3 Chunk Scores:")

for i, (document, score) in enumerate(results, 1):
    print(f"Chunk {i} Score: {score}")

context = "\n\n".join(
    document.page_content
    for document, score in results
)


class Answer(BaseModel):
    answer: str = Field(
        description="One final answer to the question"
    )


parser = PydanticOutputParser(
    pydantic_object=Answer
)


prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Answer the question using the given context.

Give only ONE final answer.
Do not give multiple answers.
Do not mention the chunks or similarity scores.

Context:
{context}

Question:
{question}

{format_instructions}
"""
)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

chain = prompt | llm | parser

response = chain.invoke({
    "context": context,
    "question": question,
    "format_instructions": parser.get_format_instructions()
})

print("\nFinal Answer:")
print(response.answer)