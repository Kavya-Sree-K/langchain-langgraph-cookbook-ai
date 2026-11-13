from dotenv import load_dotenv
load_dotenv()

# TYPE: Structured output with with_structured_output (no parser needed)
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI


class Person(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")
    skills: list[str] = Field(description="List of skills")


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
structured_llm = llm.with_structured_output(Person)

result = structured_llm.invoke(
    "Ravi is 24 years old and knows Python, networking and SQL."
)

print(result)
print(result.name, result.age, result.skills)
