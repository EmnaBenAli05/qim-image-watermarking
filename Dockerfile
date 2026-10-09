FROM python:3.12-slim
 
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
 
WORKDIR /app
 
RUN useradd --create-home appuser && chown appuser:appuser /app
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
COPY --chown=appuser:appuser . .
USER appuser
 
ENTRYPOINT ["python", "main.py"]
CMD ["--input", "test_assets/host.png"]