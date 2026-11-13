from dotenv import load_dotenv
load_dotenv()

# TYPE: Streaming chain  (print tokens as they arrive)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chain = (
    ChatPromptTemplate.from_template("Write a short story about {topic}.")
    | llm
    | StrOutputParser()
)

for chunk in chain.stream({"topic": "a robot learning to cook"}):
    print(chunk, end="", flush=True)
print()
