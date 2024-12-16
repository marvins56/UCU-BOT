# from langchain_community.llms import LlamaCpp
# from langchain.memory import ConversationBufferMemory
# from langchain.prompts import PromptTemplate
# import os
# import logging
# from typing import Dict, Optional, List
# from langchain.chains import ConversationalRetrievalChain, LLMChain
# from langchain.tools import Tool
# from langchain.agents import initialize_agent, AgentType

# logger = logging.getLogger(__name__)

# class ModelManager:
#     _instance = None
    
    
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super().__new__(cls)
#             # Initialize models here
#         return cls._instance
#     MODELS = {
#         "mistral": {
#             "name": "Mistral 7B",
#             "path": "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#             "context_length": 2048,
#             "memory_required": "4GB",
#             "model_kwargs": {
#                 "temperature": 0.7,
#                 "max_tokens": 2000,
#                 "top_p": 0.95,
#                 "top_k": 40,
#                 "n_gpu_layers": 0,
#                 "n_batch": 32,
#                 "n_ctx": 2048,
#                 "f16_kv": True,
#                 "verbose": False,
#                 "streaming": False,
#                 "repeat_penalty": 1.1
#             }
#         },
#         "phi2": {
#             "name": "Phi-2",
#             "path": "./models/phi-2.Q4_K_M.gguf",
#             "context_length": 2048,
#             "memory_required": "2.5GB",
#             "model_kwargs": {
#                 "temperature": 0.7,
#                 "max_tokens": 2000,
#                 "top_p": 0.95,
#                 "top_k": 40,
#                 "n_gpu_layers": 0,
#                 "n_batch": 32,
#                 "n_ctx": 2048,
#                 "f16_kv": True,
#                 "verbose": False,
#                 "streaming": False,
#                 "repeat_penalty": 1.1
#             }
#         }
#     }

#     def __init__(self):
#         self.current_model = None
#         self.qa_chain = None
#         self.memory = None
#         self.loaded_models = {}  # Cache for loaded models
#         self.vectorstore = None
#         self._initialize_memory()
#         self.tools = self._create_tools()
#         self.general_chain = None
#         self._initialize_chains()
#     def _create_tools(self) -> List[Tool]:
#         return [
#             Tool(
#                 name="University Knowledge Base",
#                 func=self._query_vector_store,
#                 description="Useful for questions about the university, education, courses, and academic matters"
#             ),
#             Tool(
#                 name="General Conversation",
#                 func=self._general_conversation,
#                 description="Use for general conversation, greetings, and non-academic questions"
#             )
#         ]

#     def _initialize_memory(self):
#         self.memory = ConversationBufferMemory(
#             memory_key="chat_history",
#             output_key="answer",
#             return_messages=True
#         )

#     def set_vectorstore(self, vectorstore):
#         """Set the vector store reference"""
#         self.vectorstore = vectorstore
#         logger.info("Vector store initialized in ModelManager")

#     def _create_model(self, model_key: str) -> Optional[LlamaCpp]:
#         if model_key in self.loaded_models:
#             logger.info(f"Using cached model: {model_key}")
#             return self.loaded_models[model_key]

#         try:
#             model_config = self.MODELS[model_key]
            
#             if not os.path.exists(model_config["path"]):
#                 logger.error(f"Model file not found: {model_config['path']}")
#                 return None

#             model = LlamaCpp(
#                 model_path=model_config["path"],
#                 **model_config["model_kwargs"]
#             )
            
#             self.loaded_models[model_key] = model
#             logger.info(f"Successfully loaded and cached model: {model_config['name']}")
#             return model

#         except Exception as e:
#             logger.error(f"Error creating model {model_key}: {str(e)}")
#             return None

# #     def initialize_chain(self, model_key: str, retriever) -> bool:
# #         try:
# #             if model_key not in self.MODELS:
# #                 logger.error(f"Invalid model key: {model_key}")
# #                 return False

# #             llm = self._create_model(model_key)
# #             if not llm:
# #                 return False

# #             prompt = PromptTemplate(
# #                 template="""Use the following information to answer the question. If you can't find the answer in the provided context, say that you don't have that specific information.

# # Context: {context}

# # Question: {question}

# # Previous conversation for context:
# # {chat_history}

# # Answer: """,
# #                 input_variables=["context", "question", "chat_history"]
# #             )

# #             self.qa_chain = ConversationalRetrievalChain.from_llm(
# #                 llm=llm,
# #                 retriever=retriever,
# #                 memory=self.memory,
# #                 return_source_documents=True,
# #                 chain_type="stuff",
# #                 get_chat_history=lambda h: h,
# #                 verbose=False,
# #                 combine_docs_chain_kwargs={
# #                     "prompt": prompt,
# #                     "document_variable_name": "context"
# #                 }
# #             )

# #             self.current_model = model_key
# #             return True

# #         except Exception as e:
# #             logger.error(f"Error initializing chain: {str(e)}")
# #             return False
#     def _initialize_chains(self):
#         if not self.current_model:
#             return
            
#         general_prompt = PromptTemplate(
#             template="""You are a helpful assistant. For general questions and conversation, provide friendly and informative responses.

# Previous chat: {chat_history}
# User: {question}
# Assistant: """,
#             input_variables=["question", "chat_history"]
#         )

#         self.general_chain = LLMChain(
#             llm=self.loaded_models[self.current_model],
#             prompt=general_prompt
#         )

#     def _search_vectorstore(self, query: str) -> Dict:
#         relevant_docs = self.vectorstore.max_marginal_relevance_search(
#             query, k=4, fetch_k=20, lambda_mult=0.7
#         )
#         return {
#             'content': [doc.page_content for doc in relevant_docs],
#             'metadata': [doc.metadata for doc in relevant_docs]
#         }

#     def _general_chat(self, query: str) -> str:
#         if not self.general_chain:
#             self._initialize_chains()
#         response = self.general_chain.invoke({
#             "question": query,
#             "chat_history": self.memory.chat_memory.messages if self.memory else []
#         })
#         return response["text"]

#     def _route_query(self, query: str) -> Dict:
#         # Keywords for education-related queries
#         edu_keywords = [
#     "university", "course", "study", "class", "lecture", 
#     "professor", "student", "faculty", "department", "degree", 
#     "research", "academic", "semester", "scholarship", "campus", 
#     "college", "curriculum", "major", "minor", "postgraduate", 
#     "undergraduate", "graduation", "diploma", "online course", 
#     "workshop", "internship", "thesis", "dissertation", "assignment", 
#     "examination", "graduation", "certificate", "enrollment", 
#     "classroom", "lecture hall", "campus life", "faculty member", 
#     "department head", "assistant professor", "professor emeritus", 
#     "teaching assistant", "student union", "study abroad", "study group", 
#     "research paper", "lab", "academic advisor", "student body", 
#     "student loan", "tuition", "scholar", "academic year", "coursework", 
#     "study plan", "learning", "educational institution", "college life", 
#     "degree program", "higher education", "continuing education", 
#     "online learning", "MOOC", "e-learning"
# ]

        
#         # Keywords for general conversation
#         chat_keywords = [
#     "hello", "hi", "hey", "thanks", "thank you", "how are you", 
#     "good morning", "good evening", "how's it going", "what's up", 
#     "howdy", "greetings", "yo", "hiya", "what's new", "long time no see", 
#     "nice to meet you", "pleasure", "thankful", "appreciate it", 
#     "goodbye", "see you", "take care", "bye", "talk to you later", 
#     "have a good day", "good night", "good to see you", "cheers", 
#     "thanks a lot", "welcome", "no worries", "no problem", "how have you been", 
#     "what's going on", "everything okay", "how's your day", "how's life", 
#     "what's happening", "catch you later", "let's chat", "let's talk", 
#     "what's going down", "sup", "been a while", "it's been a minute", 
#     "what's cracking", "how's everything", "any news", "any updates", 
#     "how's work", "how's school", "how's your week", "how's the weather", 
#     "what's the vibe", "feeling good", "having a great day"
# ]

        
#         query_lower = query.lower()
        
#         if any(keyword in query_lower for keyword in chat_keywords):
#             response = self._general_chat(query)
#             return {
#                 'answer': response,
#                 'source_type': 'general',
#                 'success': True
#             }
        
#         if any(keyword in query_lower for keyword in edu_keywords):
#             results = self._search_vectorstore(query)
#             return {
#                 'answer': results['content'],
#                 'sources': results['metadata'],
#                 'source_type': 'vectorstore',
#                 'success': True
#             }
            
#         # For other queries, use general chat
#         response = self._general_chat(query)
#         return {
#             'answer': response,
#             'source_type': 'general',
#             'success': True
#         }

#     # Modify existing get_response to use routing
#     def get_response(self, query: str) -> Dict:
#         try:
#             if not self.qa_chain:
#                 return {
#                     'error': 'No model initialized',
#                     'success': False
#                 }

#             response = self._route_query(query)
#             if not response['success']:
#                 return response

#             return {
#                 'answer': response['answer'],
#                 'sources': response.get('sources', []),
#                 'model': self.MODELS[self.current_model]['name'],
#                 'success': True,
#                 'source_type': response['source_type']
#             }

#         except Exception as e:
#             logger.error(f"Error getting response: {str(e)}")
#             return {
#                 'error': str(e),
#                 'success': False
#             }
#     # def get_response(self, query: str) -> Dict:
#     #     try:
#     #         if not self.qa_chain:
#     #             return {
#     #                 'error': 'No model initialized',
#     #                 'success': False
#     #             }

#     #         # Get relevant documents using MMR
#     #         relevant_docs = self.vectorstore.max_marginal_relevance_search(
#     #             query,
#     #             k=4,
#     #             fetch_k=20,
#     #             lambda_mult=0.7
#     #         )

#     #         # Get response from chain
#     #         response = self.qa_chain.invoke({
#     #             "question": query,
#     #             "chat_history": self.memory.chat_memory.messages if self.memory else []
#     #         })
            
#     #         has_relevant_info = len(response.get('source_documents', [])) > 0
            
#     #         return {
#     #             'answer': response['answer'],
#     #             'sources': [doc.metadata for doc in response['source_documents']] if has_relevant_info else [],
#     #             'model': self.MODELS[self.current_model]['name'],
#     #             'success': True,
#     #             'has_context': has_relevant_info
#     #         }

#     #     except Exception as e:
#     #         logger.error(f"Error getting response: {str(e)}")
#     #         return {
#     #             'error': str(e),
#     #             'success': False
#     #         }

#     def get_available_models(self) -> Dict:
#         return self.MODELS

#     def clear_memory(self):
#         self._initialize_memory()

#     def cleanup(self):
#         """Clean up resources"""
#         for model in self.loaded_models.values():
#             try:
#                 model.cleanup()
#             except:
#                 pass
#         self.loaded_models.clear()


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
                template="""Use the following information to answer the question. If it's a greeting or general question, respond naturally.
                For university-related questions, use the provided context. If no context is relevant, provide a general response.

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
        """Determine the type of query"""
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

        
        # Keywords for general conversation
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

    def get_response(self, query: str) -> Dict:
        try:
            if not self.qa_chain:
                return {
                    'error': 'No model initialized',
                    'success': False
                }

            query_type = self._determine_query_type(query)
            
            if query_type == "academic":
                # Search vector store for relevant documents
                relevant_docs = self.vectorstore.max_marginal_relevance_search(
                    query,
                    k=4,
                    fetch_k=20,
                    lambda_mult=0.7
                )
            else:
                relevant_docs = []

            # Get response from chain with or without context
            response = self.qa_chain.invoke({
                "question": query,
                "chat_history": self.memory.chat_memory.messages if self.memory else []
            })
            
            has_relevant_info = len(relevant_docs) > 0
            
            return {
                'answer': response['answer'],
                'sources': [doc.metadata for doc in response.get('source_documents', [])] if has_relevant_info else [],
                'model': self.MODELS[self.current_model]['name'],
                'success': True,
                'query_type': query_type,
                'has_context': has_relevant_info
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