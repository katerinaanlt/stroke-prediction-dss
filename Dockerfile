FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and dataset
COPY data/ ./data/
COPY src/ ./src/
COPY app/ ./app/

# Run preprocessing and model training during image build to generate models/stroke_pipeline.pkl
RUN python src/train.py

EXPOSE 5000

# Run production WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.app:app"]
