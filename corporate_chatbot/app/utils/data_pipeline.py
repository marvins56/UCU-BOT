from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from typing import List, Dict, Union
import chromadb
from datetime import datetime

class DataManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

    def process_text(self, text: str, metadata: Dict) -> List[Dict]:
        chunks = self.text_splitter.split_text(text)
        return [{'text': chunk, 'metadata': metadata} for chunk in chunks]

    def add_to_collection(self, collection_name: str, texts: List[str], metadatas: List[Dict]):
        collection = self.chroma_client.get_or_create_collection(collection_name)
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)

    def get_collections(self) -> List[Dict]:
        """Get all collections with their metadata"""
        try:
            collections = self.chroma_client.list_collections()
            return [
                {
                    'name': collection.name,
                    'document_count': collection.count(),
                    'last_updated': datetime.now().isoformat()  # You might want to store this in metadata
                }
                for collection in collections
            ]
        except Exception as e:
            print(f"Error getting collections: {str(e)}")
            return []

    def get_total_documents(self) -> int:
        """Get total number of documents across all collections"""
        try:
            total = 0
            for collection in self.chroma_client.list_collections():
                total += collection.count()
            return total
        except Exception as e:
            print(f"Error getting total documents: {str(e)}")
            return 0

    def get_total_collections(self) -> int:
        """Get total number of collections"""
        try:
            return len(self.chroma_client.list_collections())
        except Exception as e:
            print(f"Error getting total collections: {str(e)}")
            return 0

    def get_storage_used(self) -> str:
        """Get approximate storage used"""
        try:
            # This is a rough estimation
            collections = self.chroma_client.list_collections()
            total_size = 0
            for collection in collections:
                data = collection.get()
                # Rough estimation of embeddings size
                total_size += len(data.get('embeddings', [])) * 768 * 4  # 768-dim vectors, 4 bytes per float
            return f"{total_size / (1024 * 1024):.2f}"  # Convert to MB
        except Exception as e:
            print(f"Error calculating storage: {str(e)}")
            return "0.00"

    def create_collection(self, name: str) -> Dict:
        """Create a new collection"""
        try:
            collection = self.chroma_client.create_collection(
                name=name,
                metadata={'created_at': datetime.now().isoformat()}
            )
            return {"status": "success", "message": f"Collection {name} created successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def delete_collection(self, name: str) -> Dict:
        """Delete a collection"""
        try:
            self.chroma_client.delete_collection(name)
            return {"status": "success", "message": f"Collection {name} deleted successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def search(self, query: str, k: int = 4) -> List[Dict]:
        """Search across collections"""
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return [
                {
                    'text': doc.page_content,
                    'metadata': doc.metadata,
                    'score': getattr(doc, 'score', None)  # Include similarity score if available
                }
                for doc in results
            ]
        except Exception as e:
            print(f"Error searching: {str(e)}")
            return []