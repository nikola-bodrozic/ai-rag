# 1. Koristimo zvanični lagani Python imidž
FROM python:3.11-slim

# 2. Postavljamo radni direktorijum unutar kontejnera
WORKDIR /app

# 3. Sprečavamo Python da piše .pyc fajlove i omogućavamo direktan ispis u konzolu (real-time logovi)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Instaliramo sistemske zavisnosti (potrebne za kompajliranje nekih Python paketa ako zatreba)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Kopiramo requirements.txt i instaliramo Python pakete
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Kopiramo ostatak koda u kontejner
COPY . .

# 7. Otvaramo port 8000 na kojem će raditi naš FastAPI / LangGraph API
EXPOSE 8000

# 8. Komanda za pokretanje aplikacije (koristi uvicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]