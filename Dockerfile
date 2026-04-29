FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --default-timeout=1000 --retries=10 --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py", "--input", "test_assets/host.png"]