FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements-runtime.txt ./
RUN pip install -r requirements-runtime.txt
RUN useradd --create-home appuser
COPY --chown=appuser:appuser . .
RUN chown appuser:appuser /app
USER appuser
RUN python src/train_model.py
EXPOSE 8501
CMD ["sh", "-c", "python -m streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true --browser.gatherUsageStats=false"]
