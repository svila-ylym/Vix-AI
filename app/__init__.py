from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your_secret_key'  # Change this!
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'  # Use SQLite for simplicity
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'login'

    from .services import router_service
    app.register_blueprint(router_service.app)

    return app
