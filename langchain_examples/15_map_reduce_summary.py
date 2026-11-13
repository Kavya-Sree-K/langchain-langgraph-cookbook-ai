from dotenv import load_dotenv
load_dotenv()

# TYPE: Map-reduce summarization (summarize each chunk, then combine)
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

PDF = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.pdf"

docs = PyPDFLoader(str(PDF)).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100).split_documents(docs)
chunks = chunks[:8]  # keep it small to avoid rate limits

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
parser = StrOutputParser()

map_chain = ChatPromptTemplate.from_template("Summarize in 2 sentences:\n\n{text}") | llm | parser
reduce_chain = ChatPromptTemplate.from_template(
    "Combine these partial summaries into one short summary:\n\n{summaries}"
) | llm | parser

# MAP: .batch() runs all chunks in parallel
partials = map_chain.batch([{"text": c.page_content} for c in chunks])

# REDUCE
print(reduce_chain.invoke({"summaries": "\n".join(partials)}))
