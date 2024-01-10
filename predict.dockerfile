# Base image
FROM python:3.11-slim

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY my_mlops_project_mnj/ my_mlops_project_mnj/
COPY models/ models/ 
COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir 
RUN pip install . --no-deps --no-cache-dir 

# Entry point for prediction
ENTRYPOINT ["python", "-u", "my_mlops_project_mnj/models/predict_model.py"]
