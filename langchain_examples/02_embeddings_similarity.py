from dotenv import load_dotenv
import os
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.vectorstores.utils import cosine_similarity
from pydantic import BaseModel, Field


class Answer(BaseModel):
    answer: str = Field(description="Answer to the question")


parser = PydanticOutputParser(pydantic_object=Answer)


prompt = PromptTemplate(
    input_variables=["topic", "question"],
    template="""
Answer the question in simple language.

Topic: {topic}
Question: {question}

{format_instructions}
"""
)


api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)


chain = prompt | llm | parser


topic = input("Enter topic: ")
question = input("Enter question: ")


response = chain.invoke({
    "topic": topic,
    "question": question,
    "format_instructions": parser.get_format_instructions()
})


print("\nAnswer:", response.answer)


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)


vector1 = embeddings.embed_query(question)
vector2 = embeddings.embed_query(response.answer)


print(
    "Similarity:",
    cosine_similarity([vector1], [vector2])
)