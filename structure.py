import os

def create_project_structure():
    # Define the base directory
    base_dir = "corporate_chatbot"
    
    # Define all directories to create
    directories = [
        "app",
        "app/static/css",
        "app/static/js",
        "app/templates/admin",
        "app/templates/user",
        "app/routes"
    ]
    
    # Create directories
    for dir_path in directories:
        full_path = os.path.join(base_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
    
    # Create necessary files
    files = {
        "app/__init__.py": """
from flask import Flask
from flask_tailwind import Tailwind

app = Flask(__name__)
tailwind = Tailwind(app)

from app.routes import admin_routes, user_routes
""",
        
        "app/routes/__init__.py": "",
        
        "app/routes/admin_routes.py": """
from flask import render_template, Blueprint

admin = Blueprint('admin', __name__)

@admin.route('/admin/dashboard')
def dashboard():
    return render_template('admin/dashboard.html')

@admin.route('/admin/data-ingestion')
def data_ingestion():
    return render_template('admin/data_ingestion.html')

@admin.route('/admin/analytics')
def analytics():
    return render_template('admin/analytics.html')
""",
        
        "app/routes/user_routes.py": """
from flask import render_template, Blueprint

user = Blueprint('user', __name__)

@user.route('/')
def chat():
    return render_template('user/chat.html')
""",
        
        "app/templates/base.html": """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Corporate Chatbot</title>
    <link href="/static/css/tailwind.css" rel="stylesheet">
</head>
<body class="bg-gray-100">
    {% block content %}{% endblock %}
    <script src="/static/js/main.js"></script>
</body>
</html>
""",
        
        "app/templates/user/chat.html": """
{% extends "base.html" %}
{% block content %}
<div class="container mx-auto px-4 py-8">
    <div class="max-w-4xl mx-auto bg-white rounded-lg shadow-lg">
        <div class="h-96 p-4 overflow-y-auto" id="chat-messages">
            <!-- Chat messages will appear here -->
        </div>
        <div class="p-4 border-t">
            <form class="flex gap-4" id="chat-form">
                <input type="text" 
                       class="flex-1 p-2 border rounded-lg focus:outline-none focus:border-blue-500" 
                       placeholder="Type your message...">
                <button type="submit" 
                        class="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600">
                    Send
                </button>
            </form>
        </div>
    </div>
</div>
{% endblock %}
""",
        
        "requirements.txt": """
flask
python-dotenv
flask-tailwind
""",
        
        "run.py": """
from app import app

if __name__ == '__main__':
    app.run(debug=True)
"""
    }
    
    for file_path, content in files.items():
        full_path = os.path.join(base_dir, file_path)
        with open(full_path, 'w') as f:
            f.write(content.strip())

if __name__ == "__main__":
    create_project_structure()
    print("Project structure created successfully!")