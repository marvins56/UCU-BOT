import logging
import uuid
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
        self.logger = logging.getLogger(__name__)

    def process_text(self, text: str, metadata: Dict) -> List[Dict]:
        chunks = self.text_splitter.split_text(text)
        return [{'text': chunk, 'metadata': metadata} for chunk in chunks]

    def add_to_collection(self, collection_name: str, texts: List[str], metadatas: List[Dict]):
        """Add documents to a collection, creating it if it doesn't exist"""
        try:
            # Get or create collection
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={'created_at': datetime.now().isoformat()}
            )
            
            # Add the documents
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=[str(uuid.uuid4()) for _ in texts]  # Generate unique IDs
            )
            
            self.logger.info(f"Added {len(texts)} documents to collection {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding to collection {collection_name}: {str(e)}")
            return False

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

    def create_collection(self, collection_name: str) -> bool:
        """Create a new collection"""
        try:
            if not self.collection_exists(collection_name):
                self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={
                        'created_at': datetime.now().isoformat(),
                        'description': 'Collection for scraped web content'
                    }
                )
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error creating collection {collection_name}: {str(e)}")
            return False

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists"""
        try:
            collections = self.chroma_client.list_collections()
            return any(c.name == collection_name for c in collections)
        except Exception as e:
            self.logger.error(f"Error checking collection {collection_name}: {str(e)}")
            return False
        
    def delete_collection(self, name: str) -> Dict:
        """Delete a collection"""
        try:
            self.chroma_client.delete_collection(name)
            return {"status": "success", "message": f"Collection {name} deleted successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def search(self, query: str, collection_name: str = None, k: int = 4) -> List[Dict]:
        """Search in specific collection or across all collections"""
        try:
            results = []
            
            # If collection specified, search only that collection
            if collection_name:
                collection = self.chroma_client.get_collection(collection_name)
                query_embedding = self.embeddings.embed_query(query)
                collection_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    include=['documents', 'metadatas']
                )
                if collection_results['documents']:
                    for doc, metadata in zip(collection_results['documents'][0], collection_results['metadatas'][0]):
                        results.append({
                            'text': doc,
                            'metadata': metadata,
                            'collection': collection_name
                        })
                    
            # If no collection specified, search all collections
            else:
                collections = self.chroma_client.list_collections()
                for collection in collections:
                    query_embedding = self.embeddings.embed_query(query)
                    collection_results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=k,
                        include=['documents', 'metadatas']
                    )
                    if collection_results['documents']:
                        for doc, metadata in zip(collection_results['documents'][0], collection_results['metadatas'][0]):
                            results.append({
                                'text': doc,
                                'metadata': metadata,
                                'collection': collection.name
                            })

            # Sort results by relevance (if you have a score in the metadata)
            # results.sort(key=lambda x: x.get('metadata', {}).get('score', 0), reverse=True)
            
            return results[:k]  # Return top k results across all collections

        except Exception as e:
            self.logger.error(f"Error searching: {str(e)}")
            return []