from dotenv import load_dotenv
import os
from rag_pipline import RAG_Pipeline
from langchain_groq import ChatGroq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise SystemExit("GROQ_API_KEY is not set. Add it to your .env or export it.")

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=api_key)


pipeline = RAG_Pipeline()

space_key = "SOS"
page_title_prefix = "DevOps"
page_title = "DevOps: Production Change History"

documents = pipeline.readConfluencePagesByTitlePrefix(space_key, page_title_prefix)
#documents = pipeline.readPageFromConfluence(space_key, page_title)
print(documents)
# chunks = pipeline.documentChunker(documents)
# vector_store, retriever = pipeline.vectorStore(chunks=chunks, enableHyDE=True, llm=llm)

# question = input("Enter your question: ")

# rag_chain_lcel = pipeline.RAG_Chain(question, retriever, llm)