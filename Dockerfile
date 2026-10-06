# Streamlit dashboard.  docker build -t housing-app .  &&  docker run -p 8501:8501 housing-app
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY housing_app ./housing_app
COPY app.py ./
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY artifacts ./artifacts
RUN useradd --create-home appuser && mkdir -p /app/user_data && chown appuser /app/user_data
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
