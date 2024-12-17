import datetime
import logging
import os
import uuid
from ..utils.web_scraper import WebScraper
from ..utils.data_pipeline import DataManager
from ..utils.document_processor import DocumentProcessor
from ..utils.model_manager import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotController:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.web_scraper = WebScraper()
        self.data_manager = DataManager()
        self.model_manager = ModelManager()
        
        # Initialize vector store integration
        self.model_manager.set_vectorstore(self.data_manager.vectorstore)
        
        # Pre-load models
        self._preload_models()
        
    def _preload_models(self):
        """Pre-load all models at initialization"""
        logger.info("Pre-loading models...")
        for model_key in self.model_manager.MODELS.keys():
            self.model_manager._create_model(model_key)
        logger.info("Models pre-loaded successfully")
        
    def process_incoming_content(self, content_type, content):
        logger.info(f"Processing incoming content of type: {content_type}")
        
        if content_type == "document":
            try:
                file_path = self.save_uploaded_file(content)
                processed_content = self.document_processor.process_file(file_path)
                file_name = content.filename
                file_type = content.content_type
                
                texts = [doc.page_content for doc in processed_content['documents']]
                metadatas = [
                    {
                        'source': 'document', 
                        'file_name': file_name, 
                        'file_type': file_type,
                        'chunk_index': idx,
                        'total_chunks': len(texts)
                    } for idx, _ in enumerate(texts)
                ]
                
                logger.info("Document processed successfully")
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                return {'error': str(e), 'message': 'An error occurred while processing the document.'}
        
        elif content_type == "url":
            try:
                processed_content = self.web_scraper.scrape_url(content)
                url = content
                scrape_timestamp = datetime.datetime.now().isoformat()
                texts = [doc.page_content for doc in processed_content]
                metadatas = [
                    {
                        'source': 'url', 
                        'url': url, 
                        'scrape_timestamp': scrape_timestamp,
                        'chunk_index': idx,
                        'total_chunks': len(texts)
                    } for idx, _ in enumerate(texts)
                ]
                logger.info("URL scraped successfully")
            except Exception as e:
                logger.error(f"Error scraping URL: {str(e)}")
                return {'error': str(e), 'message': 'An error occurred while scraping the URL.'}
        
        # Add to vector store with enhanced metadata
        try:
            result = self.data_manager.add_to_collection("uploaded_documents", texts, metadatas)
            if not result:
                raise Exception("Failed to add content to vector store")
                
            logger.info(f"Added {len(texts)} chunks to vector store")
            return {
                'status': 'success', 
                'message': 'Content processed and added to the database.',
                'chunks_added': len(texts)
            }
            
        except Exception as e:
            logger.error(f"Error adding to vector store: {str(e)}")
            return {'error': str(e), 'message': 'An error occurred while storing the content.'}

    def save_uploaded_file(self, file):
        uploads_dir = 'uploads'
        os.makedirs(uploads_dir, exist_ok=True)
        
        filename = str(uuid.uuid4()) + '_' + file.filename
        file_path = os.path.join(uploads_dir, filename)
        
        try:
            file.save(file_path)
            logger.info(f"File saved successfully: {filename}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise
    
    def get_chat_response(self, query: str, model_key: str = None):
        logger.info(f"Generating chat response for query: {query}")
        
        try:
            # Initialize or switch model if needed
            if model_key and model_key != self.model_manager.current_model:
                logger.info(f"Switching to model: {model_key}")
                retriever = self.data_manager.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 4,
                        "fetch_k": 15,
                        "lambda_mult": 0.7
                    }
                )
                if not self.model_manager.initialize_chain(model_key, retriever):
                    raise Exception(f"Failed to initialize model: {model_key}")

            # Get response
            response = self.model_manager.get_response(query)
            if not response['success']:
                raise Exception(response.get('error', 'Unknown error in response generation'))

            logger.info("Chat response generated successfully")
            return response

        except Exception as e:
            logger.error(f"Error in chat response generation: {str(e)}")
            return {
                'error': str(e),
                'success': False,
                'message': 'Failed to generate response'
            }

    def cleanup(self):
        """Cleanup resources when shutting down"""
        try:
            self.model_manager.cleanup()
            logger.info("Resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")