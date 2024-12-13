from flask import render_template, Blueprint

user = Blueprint('user', __name__)

@user.route('/')
def home():
    return render_template('user/chat.html')