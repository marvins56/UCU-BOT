from typing import Dict, List
import asyncio

from corporate_chatbot.app.controllers.main_controller import ChatbotController

class CompleteDataFlow:
    def __init__(self):
        self.controller = ChatbotController()
        
    async def process_document(self, file_path: str) -> Dict:
        # Process document
        processed_content = await self.controller.process_incoming_content(
            "document", file_path
        )
        return processed_content
        
    async def process_url(self, url: str) -> Dict:
        # Scrape and process URL
        scraped_content = await self.controller.process_incoming_content(
            "url", url
        )
        return scraped_content
        
    def search_and_respond(self, query: str) -> str:
        # Get chat response
        response = self.controller.get_chat_response(query)
        return response

    def get_stats(self) -> Dict:
        # Get system stats
        return self.controller.data_manager.get_stats()