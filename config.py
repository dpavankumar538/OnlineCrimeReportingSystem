# config.py
# ---------------------------------------------------------
# Central configuration file for the Flask application.
# Keep this file private — never upload to GitHub with real credentials.
# ---------------------------------------------------------

import os



import os


class Config:
    # Flask secret key
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY")

    # MySQL database configuration
    MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.environ.get("MYSQL_PORT", 3306))
    MYSQL_USER = os.environ.get("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD")
    MYSQL_DB = os.environ.get("MYSQL_DB", "crime_db")
    MYSQL_CURSORCLASS = "DictCursor"

    # SSL for hosted Aiven MySQL
    MYSQL_CUSTOM_OPTIONS = {"ssl": {}}

    # File upload configuration
    UPLOAD_FOLDER = "static/uploads"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {
        "png", "jpg", "jpeg", "gif", "pdf", "mp4"
    }
