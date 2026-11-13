from dotenv import load_dotenv
load_dotenv()

# TYPE: Basic LCEL chain  (ChatPromptTemplate | LLM | StrOutputParser)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly tutor."),
    ("human", "Explain {topic} in simple words."),
])

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"topic": "TCP vs UDP"}))
