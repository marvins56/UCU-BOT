from datetime import timedelta
import os
from flask import Flask
from flask_cors import CORS
from .controllers.main_controller import ChatbotController
from .routes import admin_routes, user_routes

def create_app():
    app = Flask(__name__)
    app.secret_key = os.urandom(24)
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=10)
    CORS(app)  # Enable CORS for all routes
    
    # Initialize controller
    controller = ChatbotController()
    
    # Register blueprints
    app.register_blueprint(admin_routes.admin)
    app.register_blueprint(user_routes.user)
    
    return app