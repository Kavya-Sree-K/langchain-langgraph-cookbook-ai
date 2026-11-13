from dotenv import load_dotenv
import os
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.memory import ConversationBufferMemory
from pydantic import BaseModel, Field


class Answer(BaseModel):
    answer: str = Field(description="Answer to the question")
    example: str = Field(description="Simple example")


parser = PydanticOutputParser(pydantic_object=Answer)


memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True
)


prompt_template = PromptTemplate(
    input_variables=[
        "history",
        "question",
        "language",
        "level",
        "format"
    ],
    template="""
You are a helpful assistant.

Previous conversation:
{history}

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


while True:

    question = input("\nEnter your question (type exit to stop): ")

    if question.lower() == "exit":
        break

    history = memory.load_memory_variables({})["history"]

    response = chain.invoke({
        "history": history,
        "question": question,
        "language": "English",
        "level": "Beginner",
        "format": "Simple explanation with example",
        "format_instructions": parser.get_format_instructions()
    })

    print("\nAnswer:", response.answer)
    print("Example:", response.example)

    memory.save_context(
        {"input": question},
        {"output": response.answer}
    )