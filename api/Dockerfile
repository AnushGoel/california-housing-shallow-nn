# REST API. Build from the repository root:  docker build -f api/Dockerfile -t housing-api .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt
COPY housing_app ./housing_app
COPY api ./api
COPY artifacts ./artifacts
RUN useradd --create-home apiuser
USER apiuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
