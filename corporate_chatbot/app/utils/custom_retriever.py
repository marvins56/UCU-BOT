from langchain.schema import BaseRetriever, Document
from typing import List

class HybridRetriever(BaseRetriever):
    def __init__(self, data_manager, collection_name: str, k: int = 4):
        super().__init__()
        self.data_manager = data_manager
        self.collection_name = collection_name
        self.k = k

    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.data_manager.hybrid_search(
            query=query,
            collection_name=self.collection_name,
            k=self.k
        )
        
        # Convert results to LangChain Document objects
        documents = []
        for result in results:
            documents.append(
                Document(
                    page_content=result['text'],
                    metadata=result['metadata']
                )
            )
        return documents
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        # Async implementation (required by BaseRetriever)
        return self.get_relevant_documents(query)