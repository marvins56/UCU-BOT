from langchain_community.document_loaders import AsyncHtmlLoader, WebBaseLoader

from bs4 import BeautifulSoup
import urllib.parse
import asyncio
import requests
from typing import List, Set

class WebScraper:
    def __init__(self, max_depth=2, same_domain=True):
        self.max_depth = max_depth
        self.same_domain = same_domain
        self.visited_urls = set()
        self.base_domain = None

    def _get_domain(self, url: str) -> str:
        return urllib.parse.urlparse(url).netloc

    async def scrape_url(self, url: str, depth=0) -> List[dict]:
        if depth > self.max_depth or url in self.visited_urls:
            return []

        if self.same_domain:
            if not self.base_domain:
                self.base_domain = self._get_domain(url)
            elif self._get_domain(url) != self.base_domain:
                return []

        self.visited_urls.add(url)
        
        try:
            loader = AsyncHtmlLoader([url])
            docs = await loader.load()
            
            # Extract links for further crawling
            soup = BeautifulSoup(docs[0].page_content, 'html.parser')
            links = {a.get('href') for a in soup.find_all('a', href=True)}
            clean_links = {urllib.parse.urljoin(url, link) for link in links}
            
            # Clean content
            content = self._clean_content(soup)
            
            results = [{
                'url': url,
                'content': content,
                'depth': depth,
                'title': soup.title.string if soup.title else ''
            }]
            
            # Recursively scrape linked pages
            for link in clean_links:
                results.extend(await self.scrape_url(link, depth + 1))
            
            return results
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return []

    def _clean_content(self, soup: BeautifulSoup) -> str:
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'ads']):
            element.decompose()
        
        # Get main content
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        
        if main_content:
            return main_content.get_text(separator='\n', strip=True)
        return soup.get_text(separator='\n', strip=True)