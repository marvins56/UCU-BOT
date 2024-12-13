from ..utils.web_scraper import WebScraper
from ..utils.data_pipeline import DataManager
from ..utils.document_processor import DocumentProcessor


class ChatbotController:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.web_scraper = WebScraper()
        self.data_manager = DataManager()
        
    def process_incoming_content(self, content_type, content):
        # Process uploaded documents or scraped content
        if content_type == "document":
            processed_content = self.document_processor.process_file(content)
        elif content_type == "url":
            processed_content = self.web_scraper.scrape_url(content)
            
        # Store in vector database
        self.data_manager.add_to_collection("main", processed_content)
        
    def get_chat_response(self, query):
        # Search vector database for relevant content
        context = self.data_manager.search(query)
        # Generate response using context
        response = self.generate_response(query, context)
        return response