# Gebruik stabiele Python-versie
FROM python:3.11-slim

# Stel werkdirectory in
WORKDIR /app

# Installeer systeemvereisten (voor pydub + ffmpeg)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*



# Kopieer dependencies en installeer ze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopieer projectbestanden
COPY . .

# Stel poort in
EXPOSE 8080

# Start de app
ENTRYPOINT ["python", "App.py"]

# Opruimen (optioneel, maar niet dubbel)
RUN rm -rf /root/.cache /tmp/* /var/cache/*



