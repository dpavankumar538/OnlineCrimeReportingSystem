# config.py
# ---------------------------------------------------------
# Central configuration file for the Flask application.
# Keep this file private — never upload to GitHub with real credentials.
# ---------------------------------------------------------

import os

class Config:
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY")

    MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.environ.get("MYSQL_PORT", 3306))
    MYSQL_USER = os.environ.get("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD")
    MYSQL_DB = os.environ.get("MYSQL_DB", "crime_db")
    MYSQL_CURSORCLASS = "DictCursor"
    MYSQL_CUSTOM_OPTIONS = {"ssl": {}}

    # SSL for the hosted Aiven MySQL database

    UPLOAD_FOLDER = "static/uploads"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {
        "png", "jpg", "jpeg", "gif", "pdf", "mp4"
    }