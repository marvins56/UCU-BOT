from flask import Flask

app = Flask(__name__)

from app.routes.admin_routes import admin
from app.routes.user_routes import user

app.register_blueprint(admin)
app.register_blueprint(user)