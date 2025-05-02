FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
#COPY requirements.txt .
#RUN pip install --no-cache-dir -r requirements.txt \
RUN pip install fastapi \
    uvicorn \
    python-multipart

RUN pip install --extra-index-url https://download.pytorch.org/whl/cu118 transparent-background

# Copy the project files
COPY . .

# Install the package in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app

# Expose the port
EXPOSE 8000

# Default command to run the FastAPI server
CMD ["python", "app.py"]