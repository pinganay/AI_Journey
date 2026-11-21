import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from atlassian import Confluence
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from typing import List

class FileReadingAndProcessing:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"})

    def _getConfluenceClient(self) -> Confluence:
        conf_api_token = os.environ.get("CONFLUENCE_API_TOKEN")
        if not conf_api_token:
            raise SystemExit("CONFLUENCE_API_TOKEN is not set. Run: export CONFLUENCE_API_TOKEN='<your token>'")
        
        confluence = Confluence(
            url="https://elementsci.atlassian.net/wiki",  # <--- MUST include '/wiki' for Confluence Cloud
            username="rabi@elementsci.com",
            password=conf_api_token,
            cloud=True
        )
        return confluence

    def _confluence_storage_to_document(self, page: dict, space_key: str) -> Document | None:
        storage_html = (
            page.get("body", {})
            .get("storage", {})
            .get("value")
        )
        if not storage_html:
            return None

        title = page.get("title", "")
        page_id = page.get("id", "")
        text = BeautifulSoup(storage_html, "html.parser").get_text("\n", strip=True)

        return Document(
            page_content=text,
            metadata={
                "source": f"confluence:{space_key}/{title}",
                "page_id": page_id,
                "page_title": title,
                "space_key": space_key,
            },
        )

    def readConfluencePagesByTitlePrefix(self, space_key: str, title_prefix: str, batch_limit: int = 50) -> list[Document]:
        """
        Read Confluence pages whose titles start with the given prefix (case-sensitive).
        This method uses Confluence's CQL search to find pages whose titles start with the given prefix. 
        It searches in batches of 50 pages, or the number of pages passed in the limit parameter, at a time until no more matching pages are found.

        Args:
            space_key (str): The key of the Confluence space to search.
            title_prefix (str): The prefix of the page titles to match.
            batch_limit (int): The maximum number of pages to retrieve per request.

        Returns:
            list[Document]: A list of Document objects representing the matching pages.
        """

        confluence = self._getConfluenceClient()
        docs: list[Document] = []
        start = 0

        while True:
            # 1. Fetch lightweight metadata first (no body.storage)
            pages = confluence.get_all_pages_from_space(
                space=space_key,
                start=start,
                limit=batch_limit,
            )
            if not pages:
                break

            for page in pages:
                title = page.get("title", "")
                matches = title.startswith(title_prefix)
                
                if not matches:
                    continue

                # 2. Fetch full storage HTML only for matching pages
                page_id = page.get("id")
                if not page_id:
                    continue

                full_page = confluence.get_page_by_id(page_id, expand="body.storage")
                doc = self._confluence_storage_to_document(full_page, space_key)
                if doc:
                    docs.append(doc)

            if len(pages) < batch_limit:
                break
            start += batch_limit

        return docs



    def readPageFromConfluence(self, space_key, page_title):
        confluence = self._getConfluenceClient()

        page = confluence.get_page_by_title(space_key, page_title, expand="body.storage")
        if page is None:
            raise ValueError(f"Confluence page '{page_title}' not found in space '{space_key}'")

        html = page["body"]["storage"]["value"]  # page content in Confluence's HTML-like format

        text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)

        return [Document(
            page_content=text,
            metadata={
                "source": f"confluence:{space_key}/{page_title}",
                "page_title": page_title,
                "space_key": space_key,
            },
        )]


    def PDFLoader(self, pdf_paths: List[str]):
        pdf_content = []

        for pdf_path in pdf_paths:
            pdf_loader = PyMuPDFLoader(pdf_path)
            doc = pdf_loader.load()
            pdf_content.extend(doc) 

        print(f"Loaded {len(pdf_paths)} files\n")
        return pdf_content

    def semanticChunker(self, documents: List[Document]):
        chunker = SemanticChunker(self.embedding_model, number_of_chunks=15)
        chunks = chunker.split_documents(documents)
        return chunks
    
    def documentChunker(self, documents: List[Document], chunk_size=500, chunk_overlap=50):
        textSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,        
            separators=["\n", " "]
        )

        final_chunks = []
        chunks = textSplitter.split_documents(documents)
        final_chunks.extend(chunks)

        return final_chunks