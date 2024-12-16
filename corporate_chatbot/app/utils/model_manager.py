from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize models here
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
        self.loaded_models = {}  # Cache for loaded models
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
                template="""Use the following information to answer the question. If you can't find the answer in the provided context, say that you don't have that specific information.

Context: {context}

Question: {question}

Previous conversation for context:
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

    def get_response(self, query: str) -> Dict:
        try:
            if not self.qa_chain:
                return {
                    'error': 'No model initialized',
                    'success': False
                }

            # Get relevant documents using MMR
            relevant_docs = self.vectorstore.max_marginal_relevance_search(
                query,
                k=4,
                fetch_k=20,
                lambda_mult=0.7
            )

            # Get response from chain
            response = self.qa_chain.invoke({
                "question": query,
                "chat_history": self.memory.chat_memory.messages if self.memory else []
            })
            
            has_relevant_info = len(response.get('source_documents', [])) > 0
            
            return {
                'answer': response['answer'],
                'sources': [doc.metadata for doc in response['source_documents']] if has_relevant_info else [],
                'model': self.MODELS[self.current_model]['name'],
                'success': True,
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