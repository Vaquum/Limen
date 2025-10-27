FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY setup.py .
COPY loop/ loop/
COPY datasets/ datasets/
COPY docker_startup.py .

RUN pip install -e .

EXPOSE 8501

CMD ["python3", "docker_startup.py"]