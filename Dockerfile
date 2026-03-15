FROM python:3.11-slim

# Install ffmpeg and fonts
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Create working dirs
RUN mkdir -p uploads output fonts

EXPOSE 5009

CMD ["gunicorn", "--bind", "0.0.0.0:5009", "--workers", "2", "--timeout", "600", "app:app"]
