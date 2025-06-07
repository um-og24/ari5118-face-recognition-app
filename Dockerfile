FROM python:3.10-bookworm

# Set environment variables to prevent Python from writing .pyc files and to buffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Update package index and install sudo
RUN apt-get update && apt-get install -y sudo cmake libgl1 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /ari5118-face-recognition-app

# Copy all project files
COPY . .

# Make scripts executable
RUN chmod +x setup.sh start.sh

# Create virtual environment and install dependencies via setup.sh
RUN echo "🏗️ About to run setup.sh"
RUN bash -c './setup.sh -y'

# Expose ports for frontend (8501), monitoring (8502), backend (8000)
EXPOSE 8000 8501 8502

# Default command to run all components
CMD ["bash", "-c", "./start.sh --all"]