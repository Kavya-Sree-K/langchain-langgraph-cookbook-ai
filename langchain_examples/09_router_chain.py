from dotenv import load_dotenv
load_dotenv()

# TYPE: Router / branching chain  (pick a different chain based on the input)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
parser = StrOutputParser()

classifier = (
    ChatPromptTemplate.from_template(
        "Classify the question as 'technical' or 'general'. "
        "Reply with ONE word only.\n\nQuestion: {question}"
    )
    | llm
    | parser
)

technical_chain = ChatPromptTemplate.from_template(
    "You are a senior engineer. Answer in detail:\n{question}"
) | llm | parser

general_chain = ChatPromptTemplate.from_template(
    "Answer in one friendly sentence:\n{question}"
) | llm | parser

branch = RunnableBranch(
    (lambda x: "technical" in x["category"].lower(), technical_chain),
    general_chain,  # default
)

chain = RunnablePassthrough.assign(category=classifier) | branch

print(chain.invoke({"question": "How does a SIEM correlate logs?"}))
print(chain.invoke({"question": "What is the capital of France?"}))
