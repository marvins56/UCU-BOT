from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    MODELS = {
        "mistral": {
            "name": "Mistral 7B",
            "path": "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            "context_length": 2048,
            "memory_required": "4GB"
        },
        "phi2": {
            "name": "Phi-2",
            "path": "./models/phi-2.Q4_K_M.gguf",
            "context_length": 2048,
            "memory_required": "2.5GB"
        },
        "bloomz": {
            "name": "BLOOMZ-560M",
            "path": "./models/bloomz-560m.Q4_K_M.gguf",
            "context_length": 1024,
            "memory_required": "1GB"
        }
    }

    def __init__(self):
        self.current_model = None
        self.qa_chain = None
        self.memory = None
        self._initialize_memory()

    def _initialize_memory(self):
        """Initialize conversation memory"""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def _create_model(self, model_key: str) -> Optional[LlamaCpp]:
        """Create and configure the selected model"""
        try:
            model_config = self.MODELS[model_key]
            
            if not os.path.exists(model_config["path"]):
                logger.error(f"Model file not found: {model_config['path']}")
                return None

            model = LlamaCpp(
                model_path=model_config["path"],
                temperature=0.7,
                max_tokens=2000,
                n_ctx=model_config["context_length"],
                top_p=0.95,
                n_gpu_layers=0,  # CPU only
                verbose=False
            )
            
            logger.info(f"Successfully loaded model: {model_config['name']}")
            return model

        except Exception as e:
            logger.error(f"Error creating model {model_key}: {str(e)}")
            return None

    def initialize_chain(self, model_key: str, retriever) -> bool:
        """Initialize the QA chain with the selected model"""
        try:
            if model_key not in self.MODELS:
                logger.error(f"Invalid model key: {model_key}")
                return False

            llm = self._create_model(model_key)
            if not llm:
                return False

            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=False
            )

            self.current_model = model_key
            return True

        except Exception as e:
            logger.error(f"Error initializing chain: {str(e)}")
            return False

    def get_response(self, query: str) -> Dict:
        """Get response from the current model"""
        try:
            if not self.qa_chain:
                return {
                    'error': 'No model initialized',
                    'success': False
                }

            response = self.qa_chain({"question": query})
            
            return {
                'answer': response['answer'],
                'sources': [doc.metadata for doc in response['source_documents']],
                'model': self.MODELS[self.current_model]['name'],
                'success': True
            }

        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            return {
                'error': str(e),
                'success': False
            }

    def get_available_models(self) -> Dict:
        """Get list of available models and their details"""
        return self.MODELS

    def clear_memory(self):
        """Clear conversation memory"""
        self._initialize_memory()