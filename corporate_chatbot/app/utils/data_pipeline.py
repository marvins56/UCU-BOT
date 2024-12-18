

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

    def semantic_rerank(self, results: List[Dict], query: str) -> List[Dict]:
        """Improved semantic reranking with better scoring"""
        try:
            # Get query embedding once
            query_embedding = self.embeddings.embed_query(query)
            
            for result in results:
                # Get text embedding
                text_embedding = self.embeddings.embed_query(result['text'])
                
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    np.array(query_embedding).reshape(1, -1),
                    np.array(text_embedding).reshape(1, -1)
                )[0][0]
                
                # Combine with original score if exists
                original_score = result['metadata'].get('score', 0)
                result['score'] = (similarity * 0.7) + (original_score * 0.3)
            
            return sorted(results, key=lambda x: x['score'], reverse=True)
        except Exception as e:
            self.logger.error(f"Error in semantic reranking: {str(e)}")
            return results

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

    def merge_results(self, primary_results: List[Dict], secondary_results: List[Dict]) -> List[Dict]:
        seen = set()
        merged = []
        
        # First add primary results
        for result in primary_results:
            result_text = result['text']
            if result_text not in seen:
                seen.add(result_text)
                merged.append(result)
        
        # Then add unique secondary results
        for result in secondary_results:
            result_text = result['text']
            if result_text not in seen:
                seen.add(result_text)
                merged.append(result)
        
        return merged
    def search(self, query: str, collection_name: str = None, k: int = 10) -> List[Dict]:
        try:
            self.logger.info(f"Starting search for query: '{query}' in collection: {collection_name}")
            
            # First, get the query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            if collection_name:
                self.logger.info(f"Searching in specific collection: {collection_name}")
                
                # Use the collection directly through ChromaDB for more control
                collection = self.chroma_client.get_collection(collection_name)
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    include=['documents', 'metadatas', 'distances']
                )
                
                formatted_results = []
                if results['documents'] and results['documents'][0]:
                    for doc, metadata, distance in zip(
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                    ):
                        formatted_results.append({
                            'text': doc,
                            'metadata': metadata,
                            'collection': collection_name,
                            'score': 1 - distance  # Convert distance to similarity score
                        })
                    
                    self.logger.info(f"Found {len(formatted_results)} documents in {collection_name}")
                    return formatted_results
                
            else:
                self.logger.info("Searching across all collections")
                all_results = []
                collections = self.chroma_client.list_collections()
                
                for collection in collections:
                    try:
                        results = collection.query(
                            query_embeddings=[query_embedding],
                            n_results=k,
                            include=['documents', 'metadatas', 'distances']
                        )
                        
                        if results['documents'] and results['documents'][0]:
                            for doc, metadata, distance in zip(
                                results['documents'][0],
                                results['metadatas'][0],
                                results['distances'][0]
                            ):
                                all_results.append({
                                    'text': doc,
                                    'metadata': metadata,
                                    'collection': collection.name,
                                    'score': 1 - distance
                                })
                            
                            self.logger.info(f"Found {len(results['documents'][0])} documents in {collection.name}")
                    
                    except Exception as e:
                        self.logger.error(f"Error searching collection {collection.name}: {str(e)}")
                
                if all_results:
                    # Sort by score
                    all_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
                    return all_results[:k]
            
            return []

        except Exception as e:
            self.logger.error(f"Error in search: {str(e)}")
            return []
    
    
    def hybrid_search(self, query: str, collection_name: str, k: int = 10) -> List[Dict]:
        try:
            self.logger.info(f"Starting hybrid search for query: '{query}' in collection: {collection_name}")
            
            # Get vector search results
            vector_results = self.search(query, collection_name, k=k)
            self.logger.info(f"Vector search found {len(vector_results)} results")
            
            # Get keyword search results
            keyword_results = self.keyword_search(query, collection_name, k=k)
            self.logger.info(f"Keyword search found {len(keyword_results)} results")
            
            # Merge results
            combined_results = self.merge_results(vector_results, keyword_results)
            self.logger.info(f"Combined results: {len(combined_results)}")
            
            # Rerank results
            reranked_results = self.semantic_rerank(combined_results, query)[:k]
            self.logger.info(f"Final reranked results: {len(reranked_results)}")
            
            # Add context windows for top results
            enhanced_results = []
            for result in reranked_results:
                if result['score'] > 0.3:  # Lower threshold to get more results
                    context_window = self.get_context_window(result)
                    enhanced_results.append({
                        'text': context_window,
                        'metadata': result['metadata'],
                        'collection': result.get('collection', collection_name),
                        'score': result['score']
                    })
                else:
                    enhanced_results.append(result)
            
            return enhanced_results

        except Exception as e:
            self.logger.error(f"Error in hybrid search: {str(e)}")
            return []
   
    def get_context_window(self, result: Dict, window_size: int = 2) -> str:
        try:
            if not result or 'metadata' not in result:
                return result.get('text', '')

            collection = self.chroma_client.get_collection(result['collection'])
            metadata = result['metadata']
            chunk_index = metadata.get('chunk_index', 0)
            
            # Query for surrounding documents individually and combine
            context_docs = []
            
            # Get current document
            current = collection.get(
                where={"chunk_index": chunk_index}
            )
            if current['documents']:
                context_docs.extend(current['documents'])
                
            # Get previous documents
            for i in range(1, window_size + 1):
                prev_index = chunk_index - i
                if prev_index >= 0:
                    prev = collection.get(
                        where={"chunk_index": prev_index}
                    )
                    if prev['documents']:
                        context_docs = prev['documents'] + context_docs
                        
            # Get next documents
            for i in range(1, window_size + 1):
                next_index = chunk_index + i
                next_doc = collection.get(
                    where={"chunk_index": next_index}
                )
                if next_doc['documents']:
                    context_docs.extend(next_doc['documents'])
            
            if not context_docs:
                return result['text']
                
            # Combine documents into context
            context = ' '.join(context_docs)
            return context

        except Exception as e:
            self.logger.error(f"Error getting context window: {str(e)}")
            return result.get('text', '')
    def add_to_collection(self, collection_name: str, texts: List[str], metadatas: List[Dict]):
        try:
            self.logger.info(f"Adding {len(texts)} documents to collection {collection_name}")
            
            # Log first few characters of each text for debugging
            for i, text in enumerate(texts[:3]):  # Log first 3 documents
                self.logger.debug(f"Document {i} preview: {text[:100]}...")
            
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={'created_at': datetime.now().isoformat()}
            )
            
            # Add documents to Chroma
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=[str(uuid.uuid4()) for _ in texts]
            )
            
            # Verify documents were added
            count = collection.count()
            self.logger.info(f"Collection {collection_name} now has {count} documents")
            
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


    def diagnose_collection(self, collection_name: str) -> Dict:
        """Diagnostic method to check collection status and content"""
        try:
            self.logger.info(f"Diagnosing collection: {collection_name}")
            
            if not self.collection_exists(collection_name):
                return {
                    "status": "error",
                    "message": f"Collection {collection_name} does not exist",
                    "exists": False
                }
                
            collection = self.chroma_client.get_collection(collection_name)
            count = collection.count()
            
            # Get sample documents
            sample = collection.get(limit=3)
            
            return {
                "status": "success",
                "exists": True,
                "document_count": count,
                "sample_documents": sample['documents'] if count > 0 else [],
                "sample_metadata": sample['metadatas'] if count > 0 else [],
                "embedding_function": str(self.embeddings),
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            self.logger.error(f"Error diagnosing collection: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "exists": False
            }
    