from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List

class FileReadingAndProcessing:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    def PDFLoader(self, pdf_paths: List[str]):
        pdf_content = []

        for pdf_path in pdf_paths:
            pdf_loader = PyMuPDFLoader(pdf_path)
            doc = pdf_loader.load()
            pdf_content.extend(doc) 

        print(f"Loaded {len(pdf_paths)} files\n")
        return pdf_content

    def semanticChunker(self, documents: List[Document]):
        chunker = SemanticChunker(self.embedding_model)
        chunks = chunker.split_documents(documents)
        return chunks
    
    def documentChunker(self, documents: List[Document], chunk_size=50, chunk_overlap=20):
        textSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,        
            separators=["\n", " "]
        )

        final_chunks = []
        chunks = textSplitter.split_documents(documents)
        final_chunks.extend(chunks)

        return final_chunks