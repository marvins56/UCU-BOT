from io import StringIO
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
    total_urls = len(urls)
    logger.info(f"Starting scraping process for {len(urls)} URLs")

    def generate():
        for i, url in enumerate(urls):
            logger.info(f"Starting to scrape: {url}")
            progress_data = {
                'type': 'progress',
                'percentage': (i / total_urls) * 100,
                'message': f'Processing {url}'
            }
            logger.info(f"Progress: {progress_data}")
            yield f"data: {json.dumps(progress_data)}\n\n"
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(scraper.scrape_url(url))
                
                for result in results:
                    logger.info(f"Found content at: {result['url']}")
                    yield f"data: {json.dumps({'type': 'url', 'url': result['url']})}\n\n"
                    
            except Exception as e:
                error_msg = f"Error scraping {url}: {str(e)}"
                logger.error(error_msg)
                yield f"data: {json.dumps({'type': 'error', 'url': url, 'message': error_msg})}\n\n"

        completion_msg = "Scraping completed"
        logger.info(completion_msg)
        yield f"data: {json.dumps({'type': 'progress', 'percentage': 100, 'message': completion_msg})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream'
    )

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
@admin.route('/admin/scrape-logs')
def get_scrape_logs():
    # Create a handler that stores logs in memory
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Get the logger and add the handler
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    
    return jsonify({'logs': log_stream.getvalue().split('\n')})
# Error handlers
@admin.errorhandler(Exception)
def handle_error(error):
    response = {
        'error': str(error),
        'message': 'An error occurred while processing your request.'
    }
    return jsonify(response), 500