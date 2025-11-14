FROM python:3.11-slim

WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code and model files
COPY src/ src/
COPY models/ models/

# Expose the port
EXPOSE 8000

# Run the app via gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "src.app:app"]
