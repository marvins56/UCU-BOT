from io import StringIO
import os
from flask import Blueprint, render_template, request, Response, stream_with_context, jsonify
import json
import asyncio
import logging

from ..utils.web_scraper import WebScraper
from ..utils.data_pipeline import DataManager
from ..controllers.main_controller import ChatbotController

admin = Blueprint('admin', __name__)
data_manager = DataManager()
controller = ChatbotController()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@admin.route('/admin/manage')
def manage():
    return render_template('admin/manage.html')

@admin.route('/admin/upload')
def upload_page():
    return render_template('admin/upload.html')

@admin.route('/admin/scraper')
def scraper_page():
    return render_template('admin/scraper.html')

@admin.route('/admin/dashboard')
def dashboard():
    return render_template('admin/dashboard.html')

@admin.route('/admin/analytics')
def analytics():
    return render_template('admin/analytics.html')

@admin.route('/admin/scrape', methods=['POST'])
def scrape():
    data = request.get_json()
    urls = data.get('urls', [])
    depth = int(data.get('depth', 2))
    same_domain = data.get('same_domain', True)

    scraper = WebScraper(max_depth=depth, same_domain=same_domain)

    async def scrape_all_urls():
        results = []
        total_links = 0
        total_chars = 0
        
        for url in urls:
            try:
                url_results = await scraper.scrape_url(url)
                results.extend(url_results)
                
                # Calculate stats
                for result in url_results:
                    total_chars += len(result['content'])
                    total_links += result['metadata']['links_found']
                    
            except Exception as e:
                logger.error(f"Error scraping {url}: {str(e)}")
                
        return {
            'results': results,
            'stats': {
                'total_pages': len(results),
                'total_links': total_links,
                'total_chars': total_chars
            }
        }

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        scrape_data = loop.run_until_complete(scrape_all_urls())
        loop.close()

        return jsonify({
            'status': 'success',
            'results': scrape_data['results'],
            'stats': scrape_data['stats'],
            'logs': scraper.get_logs()
        })

    except Exception as e:
        logger.error(f"Scraping error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'logs': scraper.get_logs()
        }), 500

    finally:
        if 'loop' in locals() and not loop.is_closed():
            loop.close()

async def collect_results(async_gen):
    """Helper function to collect async generator results"""
    results = []
    async for item in async_gen:
        results.append(item)
    return results

@admin.route('/admin/stats')
def get_stats():
    return jsonify({
        'total_documents': data_manager.get_total_documents(),
        'total_collections': data_manager.get_total_collections(),
        'storage_used': data_manager.get_storage_used()
    })

@admin.route('/admin/collections', methods=['GET'])
def list_collections():
    collections = data_manager.get_collections()
    return jsonify(collections)

@admin.route('/admin/collections', methods=['POST'])
def create_collection():
    name = request.json.get('name')
    data_manager.create_collection(name)
    return jsonify({'message': 'Collection created successfully'})

@admin.route('/admin/collections/<name>', methods=['DELETE'])
def delete_collection(name):
    data_manager.delete_collection(name)
    return jsonify({'message': 'Collection deleted successfully'})

@admin.route('/admin/search', methods=['POST'])
def search():
    query = request.json.get('query')
    results = data_manager.search(query)
    return jsonify(results)

@admin.route('/admin/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    result = controller.process_incoming_content("document", file)
    return jsonify(result)


@admin.route('/admin/read-logs')
def read_logs():
    try:
        # Get the most recent log file from the logs directory
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        log_files = [f for f in os.listdir(logs_dir) if f.startswith('scraping_')]
        
        if not log_files:
            return jsonify({'logs': []})
            
        # Get most recent log file
        latest_log = max(log_files, key=lambda x: os.path.getctime(os.path.join(logs_dir, x)))
        log_path = os.path.join(logs_dir, latest_log)
        
        # Read logs from file
        with open(log_path, 'r', encoding='utf-8') as f:
            logs = f.readlines()
            
        # Clean and format logs
        formatted_logs = []
        for log in logs:
            if log.strip():  # Skip empty lines
                formatted_logs.append(log.strip())
        
        return jsonify({'logs': formatted_logs})
        
    except Exception as e:
        logger.error(f"Error reading logs: {str(e)}")
        return jsonify({'logs': [], 'error': str(e)})

@admin.errorhandler(Exception)
def handle_error(error):
    response = {
        'error': str(error),
        'message': 'An error occurred while processing your request.'
    }
    return jsonify(response), 500