from flask import render_template, Blueprint, jsonify, request
from ..controllers.main_controller import ChatbotController

user = Blueprint('user', __name__)
controller = ChatbotController()

@user.route('/')
def home():
    return render_template('user/chat.html')

@user.route('/chat', methods=['POST'])
def chat():
    query = request.json['message']
    response = controller.get_chat_response(query)
    return jsonify({'response': response})