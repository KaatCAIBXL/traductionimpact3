# Gebruik stabiele Python-versie
FROM python:3.11-slim

WORKDIR /app

# Installeer systeemvereisten
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Installeer Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Kopieer projectbestanden
COPY . .

EXPOSE 8080

ENTRYPOINT ["python", "App.py"]

RUN rm -rf /tmp/* /var/cache/*



RUN rm -rf /root/.cache /tmp/* /var/cache/*

# Gebruik stabiele Python-versie
FROM python:3.11-slim

WORKDIR /app

# Installeer systeemvereisten
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Installeer Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Kopieer projectbestanden
COPY . .

EXPOSE 8080

ENTRYPOINT ["python", "App.py"]

RUN rm -rf /tmp/* /var/cache/*



RUN rm -rf /root/.cache /tmp/* /var/cache/*

# Run the app (via gunicorn, if je dat gebruikt)
CMD ["gunicorn", "App:app", "--bind", "0.0.0.0:8080"]

