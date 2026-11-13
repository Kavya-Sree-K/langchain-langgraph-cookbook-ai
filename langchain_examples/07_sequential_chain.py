from dotenv import load_dotenv
load_dotenv()

# TYPE: Sequential chain  (output of chain 1 feeds chain 2)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
parser = StrOutputParser()

explain_prompt = ChatPromptTemplate.from_template("Explain {topic} in 5 sentences.")
summary_prompt = ChatPromptTemplate.from_template(
    "Summarize the following in ONE sentence:\n\n{explanation}"
)

explain_chain = explain_prompt | llm | parser

# the dict maps chain-1 output to the {explanation} variable of prompt 2
chain = {"explanation": explain_chain} | summary_prompt | llm | parser

print(chain.invoke({"topic": "Kubernetes"}))
