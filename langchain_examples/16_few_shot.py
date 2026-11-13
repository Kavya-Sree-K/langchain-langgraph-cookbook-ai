from dotenv import load_dotenv
load_dotenv()

# TYPE: Few-shot prompting chain
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

examples = [
    {"input": "I love this phone!", "output": "positive"},
    {"input": "Battery dies in an hour.", "output": "negative"},
    {"input": "It arrived on Tuesday.", "output": "neutral"},
]

example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
few_shot = FewShotChatMessagePromptTemplate(examples=examples, example_prompt=example_prompt)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify sentiment as positive, negative or neutral. One word only."),
    few_shot,
    ("human", "{input}"),
])

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
chain = prompt | llm | StrOutputParser()

print(chain.invoke({"input": "The screen is gorgeous but the speaker is awful."}))
