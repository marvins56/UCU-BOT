from flask import render_template, Blueprint, jsonify, request
from datetime import datetime
from ..utils.model_manager import ModelManager
from ..controllers.main_controller import ChatbotController
import logging
import atexit

logger = logging.getLogger(__name__)

user = Blueprint('user', __name__)
controller = ChatbotController()
model_manager = ModelManager()

# Register cleanup
@atexit.register
def cleanup():
    controller.cleanup()
    model_manager.cleanup()

@user.route('/')
def home():
    try:
        models = model_manager.get_available_models()
        return render_template(
            'user/chat.html', 
            models=models,
            now=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error loading home page: {str(e)}")
        return jsonify({'error': str(e)}), 500

@user.route('/init-model', methods=['POST'])
def init_model():
    try:
        data = request.json
        selected_model = data.get('model')

        if not selected_model:
            return jsonify({
                'error': 'No model specified',
                'success': False
            }), 400

        if model_manager.current_model != selected_model:
            logger.info(f"Initializing model: {selected_model}")
            
            # Configure MMR retriever
            retriever = controller.data_manager.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,  # Number of documents to retrieve
                    "fetch_k": 20,  # Number of documents to fetch before filtering
                    "lambda_mult": 0.7,  # Balance between relevance and diversity
                }
            )
            
            success = model_manager.initialize_chain(
                selected_model,
                retriever
            )
            
            if not success:
                return jsonify({
                    'error': f'Failed to initialize {selected_model} model',
                    'success': False
                }), 500

        return jsonify({
            'success': True,
            'model': selected_model,
            'message': f'Successfully initialized {selected_model}'
        })

    except Exception as e:
        logger.error(f"Error in model initialization: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@user.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        query = data.get('message')
        selected_model = data.get('model', 'mistral')  # Default to mistral

        if not query:
            return jsonify({
                'error': 'No message provided',
                'success': False
            }), 400

        # Get response using the controller
        response = controller.get_chat_response(query, selected_model)
        
        if not response.get('success', False):
            return jsonify({
                'error': response.get('error', 'Unknown error'),
                'success': False,
                'details': response.get('message', 'Response generation failed')
            }), 500

        # Add timestamp to response
        response['timestamp'] = datetime.now().strftime('%I:%M %p')
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False,
            'details': 'Unexpected error occurred'
        }), 500

@user.route('/clear-chat', methods=['POST'])
def clear_chat():
    try:
        model_manager.clear_memory()
        return jsonify({
            'success': True,
            'message': 'Chat history cleared successfully',
            'timestamp': datetime.now().strftime('%I:%M %p')
        })
    except Exception as e:
        logger.error(f"Error clearing chat: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@user.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Unhandled error: {str(error)}")
    response = {
        'error': str(error),
        'success': False,
        'message': 'An unexpected error occurred'
    }
    return jsonify(response), 500