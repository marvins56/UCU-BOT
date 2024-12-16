# import logging
# import uuid
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from typing import List, Dict, Union
# import chromadb
# from datetime import datetime

# class DataManager:
#     def __init__(self, persist_directory="./chroma_db"):
#         self.persist_directory = persist_directory
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name="all-MiniLM-L6-v2"
#         )
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
#         self.chroma_client = chromadb.PersistentClient(path=persist_directory)
#         self.vectorstore = Chroma(
#             persist_directory=persist_directory,
#             embedding_function=self.embeddings
#         )
#         self.logger = logging.getLogger(__name__)

#     def process_text(self, text: str, metadata: Dict) -> List[Dict]:
#         chunks = self.text_splitter.split_text(text)
#         return [{'text': chunk, 'metadata': metadata} for chunk in chunks]

#     def add_to_collection(self, collection_name: str, texts: List[str], metadatas: List[Dict]):
#         """Add documents to a collection, creating it if it doesn't exist"""
#         try:
#             # Get or create collection
#             collection = self.chroma_client.get_or_create_collection(
#                 name=collection_name,
#                 metadata={'created_at': datetime.now().isoformat()}
#             )
            
#             # Add the documents
#             collection.add(
#                 documents=texts,
#                 metadatas=metadatas,
#                 ids=[str(uuid.uuid4()) for _ in texts]  # Generate unique IDs
#             )
            
#             self.logger.info(f"Added {len(texts)} documents to collection {collection_name}")
#             return True
            
#         except Exception as e:
#             self.logger.error(f"Error adding to collection {collection_name}: {str(e)}")
#             return False

#     def get_collections(self) -> List[Dict]:
#         """Get all collections with their metadata"""
#         try:
#             collections = self.chroma_client.list_collections()
#             return [
#                 {
#                     'name': collection.name,
#                     'document_count': collection.count(),
#                     'last_updated': datetime.now().isoformat()  # You might want to store this in metadata
#                 }
#                 for collection in collections
#             ]
#         except Exception as e:
#             print(f"Error getting collections: {str(e)}")
#             return []

#     def get_total_documents(self) -> int:
#         """Get total number of documents across all collections"""
#         try:
#             total = 0
#             for collection in self.chroma_client.list_collections():
#                 total += collection.count()
#             return total
#         except Exception as e:
#             print(f"Error getting total documents: {str(e)}")
#             return 0

#     def get_total_collections(self) -> int:
#         """Get total number of collections"""
#         try:
#             return len(self.chroma_client.list_collections())
#         except Exception as e:
#             print(f"Error getting total collections: {str(e)}")
#             return 0

#     def get_storage_used(self) -> str:
#         """Get storage used based on documents"""
#         try:
#             collections = self.chroma_client.list_collections()
#             total_size = 0
#             total_docs = 0
            
#             for collection in collections:
#                 # Get collection data
#                 data = collection.get()
#                 documents = data.get('documents', [])
#                 metadatas = data.get('metadatas', [])
                
#                 # Calculate size of documents
#                 for doc in documents:
#                     if doc:
#                         total_size += len(doc.encode('utf-8'))  # Size of text content
                
#                 # Calculate size of metadata
#                 for metadata in metadatas:
#                     if metadata:
#                         total_size += len(str(metadata).encode('utf-8'))  # Size of metadata
                
#                 total_docs += len(documents)
                
#                 # Add embedding size (768-dimensional float vectors)
#                 total_size += len(documents) * 768 * 4  # 4 bytes per float
            
#             # Convert to MB with 2 decimal places
#             size_in_mb = total_size / (1024 * 1024)
#             return f"{size_in_mb:.2f}"

#         except Exception as e:
#             self.logger.error(f"Error calculating storage: {str(e)}")
#             return "0.00"

#     def create_collection(self, collection_name: str) -> bool:
#         """Create a new collection"""
#         try:
#             if not self.collection_exists(collection_name):
#                 self.chroma_client.create_collection(
#                     name=collection_name,
#                     metadata={
#                         'created_at': datetime.now().isoformat(),
#                         'description': 'Collection for scraped web content'
#                     }
#                 )
#                 return True
#             return False
#         except Exception as e:
#             self.logger.error(f"Error creating collection {collection_name}: {str(e)}")
#             return False

#     def collection_exists(self, collection_name: str) -> bool:
#         """Check if a collection exists"""
#         try:
#             collections = self.chroma_client.list_collections()
#             return any(c.name == collection_name for c in collections)
#         except Exception as e:
#             self.logger.error(f"Error checking collection {collection_name}: {str(e)}")
#             return False
        
#     def delete_collection(self, name: str) -> Dict:
#         """Delete a collection"""
#         try:
#             self.chroma_client.delete_collection(name)
#             return {"status": "success", "message": f"Collection {name} deleted successfully"}
#         except Exception as e:
#             return {"status": "error", "message": str(e)}

#     def search(self, query: str, collection_name: str = None, k: int = 10) -> List[Dict]:
#         """Search in specific collection or across all collections"""
#         try:
#             results = []
            
#             # If collection specified, search only that collection
#             if collection_name:
#                 collection = self.chroma_client.get_collection(collection_name)
#                 query_embedding = self.embeddings.embed_query(query)
#                 collection_results = collection.query(
#                     query_embeddings=[query_embedding],
#                     n_results=k,
#                     include=['documents', 'metadatas']
#                 )
#                 if collection_results['documents']:
#                     for doc, metadata in zip(collection_results['documents'][0], collection_results['metadatas'][0]):
#                         results.append({
#                             'text': doc,
#                             'metadata': metadata,
#                             'collection': collection_name
#                         })
                    
#             # If no collection specified, search all collections
#             else:
#                 collections = self.chroma_client.list_collections()
#                 for collection in collections:
#                     query_embedding = self.embeddings.embed_query(query)
#                     collection_results = collection.query(
#                         query_embeddings=[query_embedding],
#                         n_results=k,
#                         include=['documents', 'metadatas']
#                     )
#                     if collection_results['documents']:
#                         for doc, metadata in zip(collection_results['documents'][0], collection_results['metadatas'][0]):
#                             results.append({
#                                 'text': doc,
#                                 'metadata': metadata,
#                                 'collection': collection.name
#                             })

#             # Sort results by relevance (if you have a score in the metadata)
#             results.sort(key=lambda x: x.get('metadata', {}).get('score', 0), reverse=True)
            
#             return results[:k]  # Return top k results across all collections

#         except Exception as e:
#             self.logger.error(f"Error searching: {str(e)}")
#             return []

import logging
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List, Dict, Union
import chromadb
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DataManager:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " "],
            length_function=len
        )
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        self.logger = logging.getLogger(__name__)

    def process_text(self, text: str, metadata: Dict) -> List[Dict]:
        chunks = self.text_splitter.split_text(text)
        return [{'text': chunk, 'metadata': {**metadata, 'chunk_index': i}} for i, chunk in enumerate(chunks)]

    def calculate_semantic_similarity(self, query: str, text: str) -> float:
        query_embedding = self.embeddings.embed_query(query)
        text_embedding = self.embeddings.embed_query(text)
        similarity = cosine_similarity(
            np.array(query_embedding).reshape(1, -1),
            np.array(text_embedding).reshape(1, -1)
        )
        return float(similarity[0][0])

    def rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        for result in results:
            similarity = self.calculate_semantic_similarity(query, result['text'])
            result['score'] = similarity
        return sorted(results, key=lambda x: x['score'], reverse=True)

    def keyword_search(self, query: str, collection_name: str, k: int = 10) -> List[Dict]:
        try:
            collection = self.chroma_client.get_collection(collection_name)
            results = collection.query(
                query_texts=[query],
                n_results=k,
                include=['documents', 'metadatas']
            )
            
            formatted_results = []
            if results['documents']:
                for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    formatted_results.append({
                        'text': doc,
                        'metadata': metadata,
                        'collection': collection_name
                    })
            return formatted_results
        except Exception as e:
            self.logger.error(f"Error in keyword search: {str(e)}")
            return []

    def merge_results(self, semantic_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        seen = set()
        merged = []
        
        for result in semantic_results + keyword_results:
            result_text = result['text']
            if result_text not in seen:
                seen.add(result_text)
                merged.append(result)
        
        return merged

    def hybrid_search(self, query: str, collection_name: str, k: int = 10) -> List[Dict]:
        semantic_results = self.search(query, collection_name, k=k)
        keyword_results = self.keyword_search(query, collection_name, k=k)
        combined_results = self.merge_results(semantic_results, keyword_results)
        return self.rerank_results(combined_results, query)[:k]

    def get_context_window(self, result: Dict, window_size: int = 2) -> str:
        try:
            collection = self.chroma_client.get_collection(result['collection'])
            chunk_index = result['metadata']['chunk_index']
            
            context_results = collection.get(
                where={"chunk_index": {"$gte": chunk_index - window_size, "$lte": chunk_index + window_size}}
            )
            return " ".join(context_results['documents'])
        except Exception as e:
            self.logger.error(f"Error getting context window: {str(e)}")
            return result['text']

    def search(self, query: str, collection_name: str = None, k: int = 10, fetch_k: int = 20) -> List[Dict]:
        try:
            results = []
            query_embedding = self.embeddings.embed_query(query)
            
            if collection_name:
                collection = self.chroma_client.get_collection(collection_name)
                collection_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    fetch_k=fetch_k,
                    include=['documents', 'metadatas', 'distances'],
                    mmr=True,
                    mmr_lambda=0.7
                )
                
                if collection_results['documents']:
                    for doc, metadata, distance in zip(
                        collection_results['documents'][0],
                        collection_results['metadatas'][0],
                        collection_results['distances'][0]
                    ):
                        results.append({
                            'text': doc,
                            'metadata': {**metadata, 'score': 1 - distance},
                            'collection': collection_name
                        })
            else:
                collections = self.chroma_client.list_collections()
                for collection in collections:
                    collection_results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=k,
                        fetch_k=fetch_k,
                        include=['documents', 'metadatas', 'distances'],
                        mmr=True,
                        mmr_lambda=0.7
                    )
                    
                    if collection_results['documents']:
                        for doc, metadata, distance in zip(
                            collection_results['documents'][0],
                            collection_results['metadatas'][0],
                            collection_results['distances'][0]
                        ):
                            results.append({
                                'text': doc,
                                'metadata': {**metadata, 'score': 1 - distance},
                                'collection': collection.name
                            })

            results = self.rerank_results(results, query)
            return results[:k]

        except Exception as e:
            self.logger.error(f"Error searching: {str(e)}")
            return []

    def add_to_collection(self, collection_name: str, texts: List[str], metadatas: List[Dict]):
        try:
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={'created_at': datetime.now().isoformat()}
            )
            
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=[str(uuid.uuid4()) for _ in texts]
            )
            
            self.logger.info(f"Added {len(texts)} documents to collection {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding to collection {collection_name}: {str(e)}")
            return False

    def get_collections(self) -> List[Dict]:
        try:
            collections = self.chroma_client.list_collections()
            return [
                {
                    'name': collection.name,
                    'document_count': collection.count(),
                    'last_updated': datetime.now().isoformat()
                }
                for collection in collections
            ]
        except Exception as e:
            self.logger.error(f"Error getting collections: {str(e)}")
            return []

    def get_total_documents(self) -> int:
        try:
            total = 0
            for collection in self.chroma_client.list_collections():
                total += collection.count()
            return total
        except Exception as e:
            self.logger.error(f"Error getting total documents: {str(e)}")
            return 0

    def get_total_collections(self) -> int:
        try:
            return len(self.chroma_client.list_collections())
        except Exception as e:
            self.logger.error(f"Error getting total collections: {str(e)}")
            return 0

    def get_storage_used(self) -> str:
        try:
            collections = self.chroma_client.list_collections()
            total_size = 0
            
            for collection in collections:
                data = collection.get()
                documents = data.get('documents', [])
                metadatas = data.get('metadatas', [])
                
                for doc in documents:
                    if doc:
                        total_size += len(doc.encode('utf-8'))
                
                for metadata in metadatas:
                    if metadata:
                        total_size += len(str(metadata).encode('utf-8'))
                
                total_size += len(documents) * 768 * 4  # For embedding vectors
            
            size_in_mb = total_size / (1024 * 1024)
            return f"{size_in_mb:.2f}"

        except Exception as e:
            self.logger.error(f"Error calculating storage: {str(e)}")
            return "0.00"

    def create_collection(self, collection_name: str) -> bool:
        try:
            if not self.collection_exists(collection_name):
                self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={
                        'created_at': datetime.now().isoformat(),
                        'description': 'Collection for vector store data'
                    }
                )
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error creating collection {collection_name}: {str(e)}")
            return False

    def collection_exists(self, collection_name: str) -> bool:
        try:
            collections = self.chroma_client.list_collections()
            return any(c.name == collection_name for c in collections)
        except Exception as e:
            self.logger.error(f"Error checking collection {collection_name}: {str(e)}")
            return False
        
    def delete_collection(self, name: str) -> Dict:
        try:
            self.chroma_client.delete_collection(name)
            return {"status": "success", "message": f"Collection {name} deleted successfully"}
        except Exception as e:
            return {"status": "error", "message": str(e)}