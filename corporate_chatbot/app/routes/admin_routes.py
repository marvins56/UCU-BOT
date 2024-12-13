from flask import Blueprint, render_template, request, Response, stream_with_context, jsonify
import json
import asyncio

from ..utils.web_scraper import WebScraper
from ..utils.data_pipeline import DataManager
from ..controllers.main_controller import ChatbotController

admin = Blueprint('admin', __name__)
data_manager = DataManager()
controller = ChatbotController()



@admin.route('/admin/dashboard')
def dashboard():
    return render_template('admin/dashboard.html')

@admin.route('/admin/data-ingestion')
def data_ingestion():
    return render_template('admin/data_ingestion.html')

@admin.route('/admin/analytics')
def analytics():
    return render_template('admin/analytics.html')



@admin.route('/admin/scrape', methods=['POST'])
async def scrape():
    data = request.get_json()
    urls = data.get('urls', [])
    depth = int(data.get('depth', 2))
    same_domain = data.get('same_domain', True)

    scraper = WebScraper(max_depth=depth, same_domain=same_domain)
    total_urls = len(urls)

    async def generate():
        for i, url in enumerate(urls):
            yield f"data: {json.dumps({'type': 'progress', 'percentage': (i / total_urls) * 100, 'message': f'Processing {url}'})}\n\n"
            
            try:
                results = await scraper.scrape_url(url)
                for result in results:
                    yield f"data: {json.dumps({'type': 'url', 'url': result['url']})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'url': url, 'message': str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'progress', 'percentage': 100, 'message': 'Scraping completed'})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream'
    )

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