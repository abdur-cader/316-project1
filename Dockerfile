FROM python:3.11-slim

# Spark (PySpark) requires Java.
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    ca-certificates \
    fonts-dejavu-core \
    libfreetype6 \
    libpng16-16 \
  && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + data. The Python script expects DATA_PATH = "MrheLandGrants.csv"
# but the dataset file in this folder is named "Mrhe_Land_Grants.csv".
COPY main.py ./
COPY Mrhe_Land_Grants.csv ./MrheLandGrants.csv

# Optional: Spark UI (only useful if you attach a browser to the container)
EXPOSE 4050

CMD ["python", "main.py"]
