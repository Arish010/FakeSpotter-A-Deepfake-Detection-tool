# Django settings for project_settings project.


import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PROJECT_DIR)

LSTM_WEIGHTS_PATH = os.path.join(PROJECT_DIR, "lstm_binary_weights.pth")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

CALIBRATED_THRESHOLD_PATH = os.path.join(PROJECT_DIR, "best_threshold.json")
UNCERTAIN_BAND = 0.05

# ------------------------------- Core -------------------------------
SECRET_KEY = "@)0qp0!&-vht7k0wyuihr+nk-b8zrvb5j^1d@vl84cd1%)f=dz"
DEBUG = True
ALLOWED_HOSTS = ["*"]

# ------------------------------- Apps / Middleware -------------------------------
INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "ml_app.apps.MlAppConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "project_settings.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [os.path.join(PROJECT_DIR, "templates")],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.messages.context_processors.messages",
                "django.template.context_processors.media",
            ],
        },
    },
]

WSGI_APPLICATION = "project_settings.wsgi.application"

# ------------------------------- Database -------------------------------
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.path.join(PROJECT_DIR, "db.sqlite3"),
    }
}

# ------------------------------- I18N -------------------------------
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = False
USE_L10N = False
USE_TZ = False

# ------------------------------- Static / Media -------------------------------
STATIC_ROOT = "/home/app/staticfiles/"
STATIC_URL = "/static/"

STATICFILES_DIRS = [
    os.path.join(PROJECT_DIR, "uploaded_images"),
    os.path.join(PROJECT_DIR, "static"),
    os.path.join(PROJECT_DIR, "models"),
]

MEDIA_URL = "/media/"
MEDIA_ROOT = os.path.join(PROJECT_DIR, "uploaded_videos")

# Upload constraints
CONTENT_TYPES = ["video"]
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB

# ------------------------------- Model/runtime knobs used by ml_app -------------------------------
FEATURE_BACKBONE = "resnet50"
INPUT_SIZE = 112
SEQ_LEN = 24
FORCE_IDX_TO_CLASS = {0: "fake", 1: "real"}
FORCE_ARGMAX_DECISION = False

DEBUG_INFER_PRINTS = True

# ------------------------------- Logging (always on) -------------------------------
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {"class": "logging.StreamHandler"},
        "file": {
            "class": "logging.FileHandler",
            "filename": os.path.join(PROJECT_DIR, "ml_app.log"),
            "mode": "a",
        },
    },
    "loggers": {
        "ml_app": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
        "django": {"handlers": ["console"], "level": "INFO", "propagate": True},
    },
}
