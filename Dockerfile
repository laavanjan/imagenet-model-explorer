FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/app/.torch

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY classifier.py app.py ./

# Pre-download the default weights so the first request isn't slow.
RUN python -c "from classifier import ImageClassifier; ImageClassifier('resnet50')"

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
