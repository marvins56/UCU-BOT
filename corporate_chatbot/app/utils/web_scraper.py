from langchain_community.document_loaders import AsyncHtmlLoader, WebBaseLoader
from bs4 import BeautifulSoup
import urllib.parse
import asyncio
import requests
import logging
from typing import List, Set, Dict
from datetime import datetime

class WebScraper:
    def __init__(self, max_depth=2, same_domain=True):
        self.max_depth = max_depth
        self.same_domain = same_domain
        self.visited_urls = set()
        self.base_domain = None
        self.logger = logging.getLogger(__name__)

    async def scrape_url(self, url: str, depth=0) -> List[Dict]:
        if depth > self.max_depth or url in self.visited_urls:
            self.logger.info(f"Skipping {url} - {'Max depth reached' if depth > self.max_depth else 'Already visited'}")
            return []

        if self.same_domain:
            if not self.base_domain:
                self.base_domain = self._get_domain(url)
                self.logger.info(f"Set base domain: {self.base_domain}")
            elif self._get_domain(url) != self.base_domain:
                self.logger.info(f"Skipping {url} - Different domain")
                return []

        self.visited_urls.add(url)
        self.logger.info(f"Scraping URL: {url} at depth {depth}")
        
        try:
            # Modified loader usage
            loader = AsyncHtmlLoader([url])
            docs = await loader.aload()  # Use aload() instead of load()
            
            if not docs:
                self.logger.warning(f"No content found for {url}")
                return []
                
            soup = BeautifulSoup(docs[0].page_content, 'html.parser')
            
            links = {a.get('href') for a in soup.find_all('a', href=True)}
            self.logger.info(f"Found {len(links)} links on {url}")
            
            clean_links = {
                urllib.parse.urljoin(url, link) 
                for link in links 
                if link and not link.startswith(('#', 'javascript:', 'mailto:'))
            }
            self.logger.info(f"Cleaned links: {len(clean_links)} valid URLs")
            
            content = self._clean_content(soup)
            self.logger.info(f"Extracted {len(content)} characters of content from {url}")
            
            results = [{
                'url': url,
                'content': content,
                'depth': depth,
                'title': soup.title.string if soup.title else '',
                'scraped_at': datetime.now().isoformat(),
                'metadata': {
                    'domain': self._get_domain(url),
                    'links_found': len(clean_links)
                }
            }]
            
            if depth < self.max_depth:
                for link in clean_links:
                    if link not in self.visited_urls:
                        self.logger.info(f"Following link: {link}")
                        results.extend(await self.scrape_url(link, depth + 1))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {str(e)}", exc_info=True)
            return []

    def _clean_content(self, soup: BeautifulSoup) -> str:
        """Clean and extract meaningful content from the page"""
        # Remove unwanted elements
        unwanted_tags = [
            'script', 'style', 'nav', 'header', 'footer', 
            'ads', 'iframe', 'noscript'
        ]
        for element in soup(unwanted_tags):
            element.decompose()
        
        # Try to find main content area
        content_priority = [
            soup.find('main'),
            soup.find('article'),
            soup.find('div', {'class': ['content', 'main-content']}),
            soup.find('body')
        ]
        
        for content_area in content_priority:
            if content_area:
                # Clean the content
                text = content_area.get_text(separator='\n', strip=True)
                # Remove excessive whitespace
                text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
                return text
                
        return soup.get_text(separator='\n', strip=True)
    

    async def get_page_metadata(self, url: str) -> Dict:
        """Get metadata about a page without full scraping"""
        try:
            loader = AsyncHtmlLoader([url])
            docs = await loader.load()
            soup = BeautifulSoup(docs[0].page_content, 'html.parser')
            
            return {
                'title': soup.title.string if soup.title else '',
                'description': soup.find('meta', {'name': 'description'})['content'] if soup.find('meta', {'name': 'description'}) else '',
                'domain': self._get_domain(url),
                'url': url
            }
        except Exception as e:
            print(f"Error getting metadata for {url}: {str(e)}")
            return {}
    def _get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urllib.parse.urlparse(url).netloc
        except Exception as e:
            self.logger.error(f"Error extracting domain from {url}: {str(e)}")
            return ""