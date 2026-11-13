from dotenv import load_dotenv
import os
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Answer(BaseModel):
    answer: str = Field(description="Answer to the question")
    example: str = Field(description="Simple example")


parser = PydanticOutputParser(pydantic_object=Answer)


prompt_template = PromptTemplate(
    input_variables=[
        "question",
        "language",
        "level",
        "format"
    ],
    template="""
Answer the following question.

Question: {question}
Language: {language}
Level: {level}
Format: {format}

{format_instructions}
"""
)



llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


chain = prompt_template | llm | parser


response = chain.invoke({
    "question": "What is Machine Learning?",
    "language": "English",
    "level": "Beginner",
    "format": "Simple explanation with example",
    "format_instructions": parser.get_format_instructions()
})


print(response)
print(response.answer)
print(response.example)