from dotenv import load_dotenv
load_dotenv()

# TYPE: Parallel chain  (RunnableParallel runs branches at the same time)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
parser = StrOutputParser()

pros_chain = ChatPromptTemplate.from_template("Give 3 short pros of {topic}.") | llm | parser
cons_chain = ChatPromptTemplate.from_template("Give 3 short cons of {topic}.") | llm | parser

parallel = RunnableParallel(pros=pros_chain, cons=cons_chain)

result = parallel.invoke({"topic": "cloud computing"})

print("PROS:\n", result["pros"])
print("\nCONS:\n", result["cons"])
