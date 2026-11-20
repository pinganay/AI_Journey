import os
from dotenv import load_dotenv
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from vector_store import VectorStore
from file_reading_processing import FileReadingAndProcessing

class RAG_Pipeline(FileReadingAndProcessing, VectorStore):
    def __init__(self):
        #Initializes parent class with all the methods and attributes of the sub classes.
        super().__init__() 

    @staticmethod
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    def RAG_Chain(self, query, retriever):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        os.environ["GROQ_API_KEY"] = api_key

        llm = ChatGroq(model="openai/gpt-oss-120b", api_key=os.getenv("GROQ_API_KEY"))

        custom_prompt = ChatPromptTemplate.from_template("""Use the following context to answer the user's query.
            If you do not know the answer based on the context, say you don't know.
            Provide specific examples from the context to support your answer.

            Context: {context}
            Question: {question}

            Answer:""")

        rag_chain_lcel = (
            {
                "context": retriever | self.format_docs,
                "question": RunnablePassthrough(),
            }
            | custom_prompt
            | llm
            | StrOutputParser()
        )

        print(f"Question: {query}")
        print("=" * 50)
        
        answer = rag_chain_lcel.invoke(query)
        print(f"Answer: {answer}")
        
        docs = retriever.invoke(query)
        print("\nSource Documents:")
        for i, doc in enumerate(docs):
            print(f"----Document {i+1}----")
            print(f"Page number: {doc.metadata.get('page', 0) + 1}")
            print(f"File path: {doc.metadata.get('source', 'Unknown')}")
            print(doc.page_content)
            print("\n")
