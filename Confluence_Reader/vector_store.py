from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class VectorStore:
    def vectorStore(self, chunks: List[Document], embedding_model=None):  
        persist_path = "./chroma_db"

        if embedding_model is None:
            ##getattr(object, "attribute_name", default_value)
            #getattr is used to retrieve the value of an attribute from an object. In this case it searches the superclass. If the attribute does not exist, it returns a default value (in this case, None).
            #If the output of getattr is None, it means that the attribute does not exist in the superclass, and we create a new embedding_model
            embedding_model = getattr(self, 'embedding_model', None) or HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            collection_name="rag_collection",
            persist_directory=persist_path,
        )

        print(f"Stored {len(chunks)} chunks in vector store\n")

        retriever = vector_store.as_retriever(
            search_kwargs={"k": 2} ## Retrieve top 2 relevant chunks
        )

        return vector_store, retriever

    def similaritySearch(self, query: str, top_k: int, vector_store: Chroma):
        results = vector_store.similarity_search_with_relevance_scores(query, top_k)

        print(f"Top {top_k} results:")
        print("="*40)
        print(f"Query: {query}")
        for i, result in enumerate(results):
            print(f"----Result {i+1}----")
            print(f"Page number: {result[0].metadata.get('page', 0) + 1}")
            print(f"File path: {result[0].metadata.get('source', 'Unknown')}")
            print(result[0].page_content)
            print(f"Similarity Score: {result[1]}")
            print("\n")