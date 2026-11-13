# LangChain & LangGraph – Types of Chains, Agents and Graphs

A learning project where **each file demonstrates one type** of LangChain chain or LangGraph graph.
All examples use **Gemini (`gemini-2.5-flash`)** and a small PDF knowledge base for RAG.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then put your Gemini key in .env
```

Get a key at https://aistudio.google.com/apikey. **Never commit `.env`** (it is in `.gitignore`).

Run any example from the project root:

```bash
python langchain_examples/06_basic_chain.py
python langgraph_examples/05_parallel_graph.py
```

## LangChain examples (`langchain_examples/`)

| File | Type |
|---|---|
| 01_prompt_parser_chain.py | Prompt \| LLM \| PydanticOutputParser (structured answer) |
| 02_embeddings_similarity.py | Chain + embeddings + cosine similarity |
| 03_memory_legacy.py | Conversation memory (`ConversationBufferMemory`, legacy) |
| 04_rag_faiss.py | RAG: PDF → chunks → FAISS → LLM (step-by-step) |
| 05_agent_search.py | Tool-calling agent with DuckDuckGo search |
| 06_basic_chain.py | Basic LCEL chain with `StrOutputParser` |
| 07_sequential_chain.py | Sequential chain (chain 1 output → chain 2) |
| 08_parallel_chain.py | Parallel chain (`RunnableParallel`) |
| 09_router_chain.py | Router / branching chain (`RunnableBranch`) |
| 10_streaming.py | Streaming tokens |
| 11_structured_output.py | `with_structured_output` |
| 12_custom_tools_agent.py | Agent with custom `@tool` functions |
| 13_chat_history.py | Modern memory (`RunnableWithMessageHistory`) |
| 14_rag_lcel.py | RAG as a pure LCEL chain with a retriever |
| 15_map_reduce_summary.py | Map-reduce summarization |
| 16_few_shot.py | Few-shot prompting |

## LangGraph examples (`langgraph_examples/`)

| File | Type |
|---|---|
| 01_sequential.py | Linear graph (plan → action → summarize) |
| 02_conditional.py | Conditional edges (branching) |
| 03_loop.py | Loop / cycle with max attempts |
| 04_research_workflow.py | Multi-node workflow with a search tool |
| 05_parallel_graph.py | Parallel fan-out / fan-in |
| 06_llm_chatbot_graph.py | LLM node with `MessagesState` |
| 07_react_agent_graph.py | Hand-built ReAct agent (`ToolNode`) |
| 08_checkpoint_memory.py | Persistent memory with checkpointer + `thread_id` |
| 09_human_in_the_loop.py | `interrupt` + resume with `Command` |
| 10_multi_agent_supervisor.py | Supervisor routing to worker agents |
| 11_agentic_rag_graph.py | Agentic RAG: retrieve → grade → rewrite/generate |

## Project structure

```
.
├── data/knowledge_base.pdf
├── langchain_examples/
├── langgraph_examples/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
