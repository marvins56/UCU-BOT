import datetime
import logging
import os
import uuid
from ..utils.web_scraper import WebScraper
from ..utils.data_pipeline import DataManager
from ..utils.document_processor import DocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotController:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.web_scraper = WebScraper()
        self.data_manager = DataManager()
        
    def process_incoming_content(self, content_type, content):
        logger.info(f"Processing incoming content of type: {content_type}")
        
        if content_type == "document":
            try:
                file_path = self.save_uploaded_file(content)
                processed_content = self.document_processor.process_file(file_path)
                file_name = content.filename
                file_type = content.content_type
                
                texts = [doc.page_content for doc in processed_content['documents']]
                metadatas = [{'source': 'document', 'file_name': file_name, 'file_type': file_type} for _ in texts]
                
                logger.info("Document processed successfully")
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                return {'error': str(e), 'message': 'An error occurred while processing the document.'}
        elif content_type == "url":
            processed_content = self.web_scraper.scrape_url(content)
            url = content
            scrape_timestamp = datetime.datetime.now().isoformat()
            metadatas = [{'source': 'url', 'url': url, 'scrape_timestamp': scrape_timestamp}]
            logger.info("URL scraped successfully")
        
        self.data_manager.add_to_collection("uploaded_documents", texts, metadatas)
        logger.info("Processed content added to the 'uploaded_documents' collection")
        
        return {'status': 'success', 'message': 'Content processed and added to the database.'}

    def save_uploaded_file(self, file):
        # Create the 'uploads' directory if it doesn't exist
        uploads_dir = 'uploads'
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Generate a unique filename to avoid conflicts
        filename = str(uuid.uuid4()) + '_' + file.filename
        file_path = os.path.join(uploads_dir, filename)
        
        # Save the file to the 'uploads' directory
        file.save(file_path)
        
        return file_path
    
    def get_chat_response(self, query):
        logger.info(f"Generating chat response for query: {query}")
        # Use hybrid search for better results
        context = self.data_manager.hybrid_search(query, collection_name="uploaded_documents")
        
        # Get context window for better understanding
        enhanced_context = []
        for result in context:
            context_window = self.data_manager.get_context_window(result)
            enhanced_context.append({
                'text': context_window,
                'metadata': result['metadata'],
                'score': result.get('score', 0)
            })
        
        logger.info("Enhanced context retrieved from the vector database")
        # Generate response using enhanced context
        response = self.generate_response(query, enhanced_context)
        logger.info("Chat response generated successfully")
        return response