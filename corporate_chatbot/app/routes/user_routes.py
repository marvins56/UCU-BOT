
from flask import render_template, Blueprint, jsonify, request

from ..utils.model_manager import ModelManager
from ..controllers.main_controller import ChatbotController
import logging

user = Blueprint('user', __name__)
controller = ChatbotController()
model_manager = ModelManager()

logger = logging.getLogger(__name__)

@user.route('/')
def home():
    # Pass available models to template
    models = model_manager.get_available_models()
    return render_template('user/chat.html', models=models)

@user.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        query = data['message']
        selected_model = data.get('model', 'mistral')  # Default to Mistral

        # Initialize chain if not already initialized or if model changed
        if model_manager.current_model != selected_model:
            success = model_manager.initialize_chain(
                selected_model,
                controller.data_manager.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 4, "fetch_k": 20}
                )
            )
            if not success:
                return jsonify({
                    'error': 'Failed to initialize model',
                    'success': False
                }), 500

        # Get response
        response = model_manager.get_response(query)
        
        if response['success']:
            return jsonify(response)
        else:
            return jsonify({
                'error': response.get('error', 'Unknown error'),
                'success': False
            }), 500

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@user.route('/clear-chat', methods=['POST'])
def clear_chat():
    try:
        model_manager.clear_memory()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500