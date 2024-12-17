
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
import os
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    MODELS = {
        "mistral": {
            "name": "Mistral 7B",
            "path": "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            "context_length": 2048,
            "memory_required": "4GB",
            "model_kwargs": {
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 0.95,
                "top_k": 40,
                "n_gpu_layers": 0,
                "n_batch": 32,
                "n_ctx": 2048,
                "f16_kv": True,
                "verbose": False,
                "streaming": False,
                "repeat_penalty": 1.1
            }
        },
        "phi2": {
            "name": "Phi-2",
            "path": "./models/phi-2.Q4_K_M.gguf",
            "context_length": 2048,
            "memory_required": "2.5GB",
            "model_kwargs": {
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 0.95,
                "top_k": 40,
                "n_gpu_layers": 0,
                "n_batch": 32,
                "n_ctx": 2048,
                "f16_kv": True,
                "verbose": False,
                "streaming": False,
                "repeat_penalty": 1.1
            }
        }
    }

    def __init__(self):
        self.current_model = None
        self.qa_chain = None
        self.memory = None
        self.loaded_models = {}
        self.vectorstore = None
        self._initialize_memory()

    def _initialize_memory(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )

    def set_vectorstore(self, vectorstore):
        """Set the vector store reference"""
        self.vectorstore = vectorstore
        logger.info("Vector store initialized in ModelManager")

    def _create_model(self, model_key: str) -> Optional[LlamaCpp]:
        if model_key in self.loaded_models:
            logger.info(f"Using cached model: {model_key}")
            return self.loaded_models[model_key]

        try:
            model_config = self.MODELS[model_key]
            
            if not os.path.exists(model_config["path"]):
                logger.error(f"Model file not found: {model_config['path']}")
                return None

            model = LlamaCpp(
                model_path=model_config["path"],
                **model_config["model_kwargs"]
            )
            
            self.loaded_models[model_key] = model
            logger.info(f"Successfully loaded and cached model: {model_config['name']}")
            return model

        except Exception as e:
            logger.error(f"Error creating model {model_key}: {str(e)}")
            return None

    def initialize_chain(self, model_key: str, retriever) -> bool:
        try:
            if model_key not in self.MODELS:
                logger.error(f"Invalid model key: {model_key}")
                return False

            llm = self._create_model(model_key)
            if not llm:
                return False

            prompt = PromptTemplate(
                template="""You are a helpful assistant specifically designed to help Uganda Christian University (UCU) students. Your role is to provide accurate and relevant information about UCU's programs, policies, campus life, and academic matters.

                Use the following information to answer the question. For UCU-specific questions, use the provided context carefully. If no UCU-specific context is available, provide a general but relevant response while acknowledging the UCU context.

                Context: {context}

                Question: {question}

                Previous conversation:
                {chat_history}

                Answer: """,
                input_variables=["context", "question", "chat_history"]
            )

            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                chain_type="stuff",
                get_chat_history=lambda h: h,
                verbose=False,
                combine_docs_chain_kwargs={
                    "prompt": prompt,
                    "document_variable_name": "context"
                }
            )

            self.current_model = model_key
            return True

        except Exception as e:
            logger.error(f"Error initializing chain: {str(e)}")
            return False
    def _determine_query_type(self, query: str) -> str:
        edu_keywords = [
            "university", "course", "study", "class", "lecture", 
            "professor", "student", "faculty", "department", "degree", 
            "research", "academic", "semester", "scholarship", "campus", 
            "college", "curriculum", "major", "minor", "postgraduate", 
            "undergraduate", "graduation", "diploma", "online course", 
            "workshop", "internship", "thesis", "dissertation", "assignment", 
            "examination", "graduation", "certificate", "enrollment", 
            "classroom", "lecture hall", "campus life", "faculty member", 
            "department head", "assistant professor", "professor emeritus", 
            "teaching assistant", "student union", "study abroad", "study group", 
            "research paper", "lab", "academic advisor", "student body", 
            "student loan", "tuition", "scholar", "academic year", "coursework", 
            "study plan", "learning", "educational institution", "college life", 
            "degree program", "higher education", "continuing education", 
            "online learning", "MOOC", "e-learning"
        ]
        
        chat_keywords = [
            "hello", "hi", "hey", "thanks", "thank you", "how are you", 
            "good morning", "good evening", "how's it going", "what's up", 
            "howdy", "greetings", "yo", "hiya", "what's new", "long time no see", 
            "nice to meet you", "pleasure", "thankful", "appreciate it", 
            "goodbye", "see you", "take care", "bye", "talk to you later", 
            "have a good day", "good night", "good to see you", "cheers", 
            "thanks a lot", "welcome", "no worries", "no problem", "how have you been", 
            "what's going on", "everything okay", "how's your day", "how's life", 
            "what's happening", "catch you later", "let's chat", "let's talk", 
            "what's going down", "sup", "been a while", "it's been a minute", 
            "what's cracking", "how's everything", "any news", "any updates", 
            "how's work", "how's school", "how's your week", "how's the weather", 
            "what's the vibe", "feeling good", "having a great day"
        ]
        
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in chat_keywords):
            return "chat"
        elif any(keyword in query_lower for keyword in edu_keywords):
            return "academic"
        return "general"

    def _hybrid_retrieval(self, query: str, k: int = 4) -> List:
        """
        Performs hybrid document retrieval using both MMR and similarity search
        """
        try:
            # Get MMR results
            mmr_docs = self.vectorstore.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=20,
                lambda_mult=0.7
            )
            
            # Get similarity search results
            similarity_docs = self.vectorstore.similarity_search(
                query,
                k=k
            )
            
            # Combine and deduplicate results based on content
            seen_contents = set()
            hybrid_results = []
            
            # Process MMR results first (giving them priority)
            for doc in mmr_docs:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    hybrid_results.append(doc)
            
            # Then add unique similarity results
            for doc in similarity_docs:
                if doc.page_content not in seen_contents and len(hybrid_results) < k:
                    seen_contents.add(doc.page_content)
                    hybrid_results.append(doc)
            
            # Score the results based on relevance to query
            scored_results = []
            if hybrid_results:
                similarity_scores = self.vectorstore.similarity_search_with_score(
                    query,
                    k=len(hybrid_results)
                )
                
                # Create a mapping of content to scores
                score_map = {doc.page_content: score for doc, score in similarity_scores}
                
                # Sort hybrid results by score
                scored_results = sorted(
                    hybrid_results,
                    key=lambda x: score_map.get(x.page_content, float('inf'))
                )
            
            return scored_results[:k]

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            return []

    # def get_response(self, query: str) -> Dict:
    #     try:
    #         if not self.qa_chain:
    #             return {
    #                 'error': 'No model initialized',
    #                 'success': False
    #             }

    #         query_type = self._determine_query_type(query)
            
    #         if query_type == "academic":
    #             relevant_docs = self._hybrid_retrieval(query, k=4)
    #         else:
    #             relevant_docs = []

    #         response = self.qa_chain.invoke({
    #             "question": query,
    #             "chat_history": self.memory.chat_memory.messages if self.memory else []
    #         })
            
    #         has_relevant_info = len(relevant_docs) > 0
            
    #         # Include confidence scores in sources if available
    #         sources = []
    #         if has_relevant_info:
    #             for doc in response.get('source_documents', []):
    #                 source_info = doc.metadata.copy()
    #                 if hasattr(doc, 'score'):
    #                     source_info['confidence_score'] = float(doc.score)
    #                 sources.append(source_info)
            
    #         return {
    #             'answer': response['answer'],
    #             'sources': sources,
    #             'model': self.MODELS[self.current_model]['name'],
    #             'success': True,
    #             'query_type': query_type,
    #             'has_context': has_relevant_info
    #         }

    #     except Exception as e:
    #         logger.error(f"Error getting response: {str(e)}")
    #         return {
    #             'error': str(e),
    #             'success': False
    #         }
    def get_response(self, query: str) -> Dict:
        try:
            if not self.qa_chain:
                return {
                    'error': 'No model initialized',
                    'success': False
                }

            query_type = self._determine_query_type(query)
            response = self.qa_chain.invoke({
                "question": query,
                "chat_history": self.memory.chat_memory.messages if self.memory else []
            })

            # Format sources with references
            sources = []
            if response.get('source_documents'):
                for doc in response['source_documents']:
                    source_info = doc.metadata.copy()
                    source_ref = {}
                    
                    # Handle different types of sources
                    if source_info.get('url'):
                        source_ref = {
                            'type': 'url',
                            'url': source_info['url'],
                            'title': source_info.get('title', 'Web Page'),
                            'timestamp': source_info.get('scrape_timestamp')
                        }
                    elif source_info.get('file_name'):
                        source_ref = {
                            'type': 'document',
                            'file_name': source_info['file_name'],
                            'file_type': source_info.get('file_type', 'Unknown'),
                            'page': source_info.get('page_number')
                        }
                    elif source_info.get('reference'):
                        source_ref = {
                            'type': 'reference',
                            'text': source_info['reference'],
                            'details': source_info.get('details', '')
                        }
                    
                    if source_ref:
                        sources.append(source_ref)

            return {
                'answer': response['answer'],
                'sources': sources,
                'model': self.MODELS[self.current_model]['name'],
                'success': True,
                'query_type': query_type,
                'has_context': bool(sources)
            }

        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            return {
                'error': str(e),
                'success': False
            }
    def get_available_models(self) -> Dict:
        return self.MODELS

    def clear_memory(self):
        self._initialize_memory()

    def cleanup(self):
        """Clean up resources"""
        for model in self.loaded_models.values():
            try:
                model.cleanup()
            except:
                pass
        self.loaded_models.clear()