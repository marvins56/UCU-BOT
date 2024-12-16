# from datetime import timedelta
# import os
# from flask import Flask
# from flask_cors import CORS
# from .controllers.main_controller import ChatbotController
# from .routes import admin_routes, user_routes
# from .utils.model_checker import check_and_download_models
# import logging


# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler('chatbot.log')
#     ]
# )
# logger = logging.getLogger(__name__)

# def create_app():
#     # Check for models before creating app
#     print("Checking for required models...")
#     if not check_and_download_models():
#         logger.error("Failed to download required models. Application cannot start.")
#         return None

#     app = Flask(__name__)
#     app.secret_key = os.urandom(24)
#     app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=10)
#     CORS(app)  # Enable CORS for all routes
    
#     # Initialize controller
#     controller = ChatbotController()
    
#     # Register blueprints
#     app.register_blueprint(admin_routes.admin)
#     app.register_blueprint(user_routes.user)
    
#     return app
from datetime import timedelta
import os
from flask import Flask
from flask_cors import CORS
from .controllers.main_controller import ChatbotController
from .routes import admin_routes, user_routes
from .utils.model_checker import check_and_download_models
import logging
import atexit

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chatbot.log')
    ]
)
logger = logging.getLogger(__name__)

def create_app():
    # Check for required models before creating app
    print("Checking for required models...")
    if not check_and_download_models():
        logger.error("Failed to download required models. Application cannot start.")
        return None

    app = Flask(__name__)
    app.secret_key = os.urandom(24)
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=10)
    
    # Enable CORS
    CORS(app)
    
    # Initialize controller
    controller = ChatbotController()
    
    # Register cleanup
    @atexit.register
    def cleanup():
        try:
            controller.cleanup()
            logger.info("Application resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    # Register blueprints
    app.register_blueprint(admin_routes.admin)
    app.register_blueprint(user_routes.user)
    
    # Store controller instance
    app.controller = controller
    
    return app