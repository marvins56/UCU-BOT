
# from langchain_community.document_loaders import AsyncHtmlLoader, WebBaseLoader
# from bs4 import BeautifulSoup
# import urllib.parse
# import asyncio
# import requests
# import logging
# from typing import List, Set, Dict
# from datetime import datetime

# class WebScraper:
#     def __init__(self, max_depth=2, same_domain=True, log_file="scraping_logs.txt"):
#         self.max_depth = max_depth
#         self.same_domain = same_domain
#         self.visited_urls = set()
#         self.base_domain = None
#         self.log_file = log_file
#         self.logger = logging.getLogger(__name__)
        
#         # Set up file handler
#         file_handler = logging.FileHandler(log_file, mode='w')
#         file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
#         self.logger.addHandler(file_handler)

#     def add_log(self, message: str, log_type: str = 'info') -> Dict:
#         """Add a log entry with timestamp"""
#         log_entry = {
#             'timestamp': datetime.now().strftime('%H:%M:%S'),
#             'message': message,
#             'type': log_type
#         }
#         self.scrape_logs.append(log_entry)
#         self.logger.info(message)  # Also log to system logger
#         return log_entry

#     def _get_domain(self, url: str) -> str:
#         """Extract domain from URL"""
#         try:
#             return urllib.parse.urlparse(url).netloc
#         except Exception as e:
#             error_msg = f"Error extracting domain from {url}: {str(e)}"
#             self.add_log(error_msg, 'error')
#             return ""

#     async def scrape_url(self, url: str, depth=0) -> List[Dict]:
#         """Main scraping function with enhanced logging and content processing"""
#         if depth > self.max_depth or url in self.visited_urls:
#             skip_reason = 'Max depth reached' if depth > self.max_depth else 'Already visited'
#             self.add_log(f"Skipping {url} - {skip_reason}")
#             return []

#         if self.same_domain:
#             if not self.base_domain:
#                 self.base_domain = self._get_domain(url)
#                 self.add_log(f"Set base domain: {self.base_domain}")
#             elif self._get_domain(url) != self.base_domain:
#                 self.add_log(f"Skipping {url} - Different domain")
#                 return []

#         self.visited_urls.add(url)
#         self.add_log(f"Scraping URL: {url} at depth {depth}")
        
#         try:
#             # Load content
#             loader = AsyncHtmlLoader([url])
#             self.add_log(f"Loading content from {url}")
#             docs = await loader.aload()
            
#             if not docs:
#                 self.add_log(f"No content found for {url}", 'warning')
#                 return []
                
#             soup = BeautifulSoup(docs[0].page_content, 'html.parser')
            
#             # Process links
#             links = {a.get('href') for a in soup.find_all('a', href=True)}
#             self.add_log(f"Found {len(links)} raw links on {url}")
            
#             clean_links = {
#                 urllib.parse.urljoin(url, link) 
#                 for link in links 
#                 if link and not link.startswith(('#', 'javascript:', 'mailto:'))
#             }
#             self.add_log(f"Processed {len(clean_links)} valid URLs")
            
#             # Extract and clean content
#             content = self._clean_content(soup)
#             content_length = len(content)
#             self.add_log(f"Extracted {content_length} characters of content from {url}")
            
#             # Create structured result
#             results = [{
#                 'url': url,
#                 'content': content,
#                 'depth': depth,
#                 'title': soup.title.string if soup.title else '',
#                 'scraped_at': datetime.now().isoformat(),
#                 'metadata': {
#                     'domain': self._get_domain(url),
#                     'links_found': len(clean_links),
#                     'content_length': content_length,
#                     'scrape_depth': depth
#                 }
#             }]
            
#             # Recursively process linked pages
#             if depth < self.max_depth:
#                 for link in clean_links:
#                     if link not in self.visited_urls:
#                         self.add_log(f"Following link: {link}")
#                         results.extend(await self.scrape_url(link, depth + 1))
            
#             return results
            
#         except Exception as e:
#             error_msg = f"Error scraping {url}: {str(e)}"
#             self.add_log(error_msg, 'error')
#             return []

#     def _clean_content(self, soup: BeautifulSoup) -> str:
#         """Enhanced content cleaning with logging"""
#         # Remove unwanted elements
#         unwanted_tags = [
#             'script', 'style', 'nav', 'header', 'footer', 
#             'ads', 'iframe', 'noscript', 'aside', 'menu'
#         ]
#         elements_removed = 0
#         for element in soup(unwanted_tags):
#             element.decompose()
#             elements_removed += 1
        
#         self.add_log(f"Removed {elements_removed} unwanted elements")
        
#         # Try to find main content area
#         content_priority = [
#             ('main', soup.find('main')),
#             ('article', soup.find('article')),
#             ('content-div', soup.find('div', {'class': ['content', 'main-content', 'post-content']})),
#             ('body', soup.find('body'))
#         ]
        
#         for area_type, content_area in content_priority:
#             if content_area:
#                 self.add_log(f"Found content in {area_type} tag")
#                 # Clean the content
#                 text = content_area.get_text(separator='\n', strip=True)
#                 # Remove excessive whitespace
#                 text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
#                 return text
        
#         self.add_log("No specific content area found, using full page content", 'warning')
#         return soup.get_text(separator='\n', strip=True)

#     async def get_page_metadata(self, url: str) -> Dict:
#         """Enhanced metadata extraction with logging"""
#         try:
#             self.add_log(f"Fetching metadata for {url}")
#             loader = AsyncHtmlLoader([url])
#             docs = await loader.aload()
#             soup = BeautifulSoup(docs[0].page_content, 'html.parser')
            
#             metadata = {
#                 'title': soup.title.string if soup.title else '',
#                 'description': soup.find('meta', {'name': 'description'})['content'] if soup.find('meta', {'name': 'description'}) else '',
#                 'domain': self._get_domain(url),
#                 'url': url,
#                 'fetched_at': datetime.now().isoformat()
#             }
            
#             self.add_log(f"Successfully extracted metadata for {url}")
#             return metadata
            
#         except Exception as e:
#             error_msg = f"Error getting metadata for {url}: {str(e)}"
#             self.add_log(error_msg, 'error')
#             return {}

#     def get_logs(self) -> List[Dict]:
#         """Return all collected logs"""
#         return self.scrape_logs

from langchain_community.document_loaders import AsyncHtmlLoader, WebBaseLoader
from bs4 import BeautifulSoup
import urllib.parse
import asyncio
import requests
import logging
import os
from typing import List, Set, Dict
from datetime import datetime

class WebScraper:
    def __init__(self, max_depth=2, same_domain=True):
        self.max_depth = max_depth
        self.same_domain = same_domain
        self.visited_urls = set()
        self.base_domain = None
        self.logger = logging.getLogger(__name__)
        self.scrape_logs = []
        self.log_file = None
        self.file_handler = None

    def _setup_logging(self):
        """Setup logging when scraping starts"""
        if not self.log_file:
            # Create logs directory if it doesn't exist
            self.logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
            os.makedirs(self.logs_dir, exist_ok=True)
            
            # Create new log file
            self.log_file = os.path.join(self.logs_dir, f'scraping_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            
            # Configure file handler
            self.file_handler = logging.FileHandler(self.log_file, mode='w')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            self.file_handler.setFormatter(formatter)
            self.logger.addHandler(self.file_handler)
            self.logger.setLevel(logging.INFO)

    def _cleanup_logging(self):
        """Cleanup logging after scraping ends"""
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)
            self.file_handler.close()
            self.file_handler = None
            self.log_file = None

    def _log(self, message: str, level: str = 'info') -> None:
        """Internal logging method"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        
        # Write to file if logging is set up
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        
        # Store in memory
        self.scrape_logs.append({
            'timestamp': timestamp,
            'message': message,
            'type': level
        })
        
        # Log using logger
        if level == 'error':
            self.logger.error(message)
        else:
            self.logger.info(message)

    def _get_domain(self, url: str) -> str:
        try:
            return urllib.parse.urlparse(url).netloc
        except Exception as e:
            self._log(f"Error extracting domain from {url}: {str(e)}", 'error')
            return ""

    async def scrape_url(self, url: str, depth=0) -> List[Dict]:
        # Setup logging on first call (depth=0)
        if depth == 0:
            self._setup_logging()
            self.scrape_logs = []  # Reset logs for new scraping session

        if depth > self.max_depth or url in self.visited_urls:
            skip_reason = 'Max depth reached' if depth > self.max_depth else 'Already visited'
            self._log(f"Skipping {url} - {skip_reason}")
            return []

        if self.same_domain:
            if not self.base_domain:
                self.base_domain = self._get_domain(url)
                self._log(f"Set base domain: {self.base_domain}")
            elif self._get_domain(url) != self.base_domain:
                self._log(f"Skipping {url} - Different domain")
                return []

        self.visited_urls.add(url)
        self._log(f"Scraping URL: {url} at depth {depth}")
        
        try:
            loader = AsyncHtmlLoader([url])
            self._log(f"Loading content from {url}")
            docs = await loader.aload()
            
            if not docs:
                self._log(f"No content found for {url}", 'warning')
                return []
                
            soup = BeautifulSoup(docs[0].page_content, 'html.parser')
            
            links = {a.get('href') for a in soup.find_all('a', href=True)}
            self._log(f"Found {len(links)} raw links on {url}")
            
            clean_links = {
                urllib.parse.urljoin(url, link) 
                for link in links 
                if link and not link.startswith(('#', 'javascript:', 'mailto:'))
            }
            self._log(f"Processed {len(clean_links)} valid URLs")
            
            content = self._clean_content(soup)
            content_length = len(content)
            self._log(f"Extracted {content_length} characters of content from {url}")
            
            results = [{
                'url': url,
                'content': content,
                'depth': depth,
                'title': soup.title.string if soup.title else '',
                'scraped_at': datetime.now().isoformat(),
                'metadata': {
                    'domain': self._get_domain(url),
                    'links_found': len(clean_links),
                    'content_length': content_length,
                    'scrape_depth': depth
                }
            }]
            
            if depth < self.max_depth:
                for link in clean_links:
                    if link not in self.visited_urls:
                        self._log(f"Following link: {link}")
                        results.extend(await self.scrape_url(link, depth + 1))
            
            # Cleanup logging when top-level scraping is done
            if depth == 0:
                self._cleanup_logging()
                
            return results
            
        except Exception as e:
            error_msg = f"Error scraping {url}: {str(e)}"
            self._log(error_msg, 'error')
            if depth == 0:
                self._cleanup_logging()
            return []

    def _clean_content(self, soup: BeautifulSoup) -> str:
        unwanted_tags = [
            'script', 'style', 'nav', 'header', 'footer', 
            'ads', 'iframe', 'noscript', 'aside', 'menu'
        ]
        elements_removed = 0
        for element in soup(unwanted_tags):
            element.decompose()
            elements_removed += 1
        
        self._log(f"Removed {elements_removed} unwanted elements")
        
        content_priority = [
            ('main', soup.find('main')),
            ('article', soup.find('article')),
            ('content-div', soup.find('div', {'class': ['content', 'main-content', 'post-content']})),
            ('body', soup.find('body'))
        ]
        
        for area_type, content_area in content_priority:
            if content_area:
                self._log(f"Found content in {area_type} tag")
                text = content_area.get_text(separator='\n', strip=True)
                text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
                return text
        
        self._log("No specific content area found, using full page content", 'warning')
        return soup.get_text(separator='\n', strip=True)

    def get_current_log_file(self) -> str:
        """Return the path to the current log file"""
        return self.log_file

    def get_logs(self) -> List[Dict]:
        """Return all collected logs"""
        return self.scrape_logs