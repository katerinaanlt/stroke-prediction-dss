FROM python:3.11-slim

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files from the current folder into the container
COPY healthcare-dataset-stroke-data.csv .
COPY preprocess.py .
COPY train.py .
COPY app.py .
COPY index.html .

# Preprocess data, evaluate models, and generate stroke_pipeline.pkl during build
RUN python train.py

EXPOSE 5000

# Start the Flask API via Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
