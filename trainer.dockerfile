# Base image
FROM python:3.11-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY my_mlops_project_mnj/ my_mlops_project_mnj/
COPY data/ data/

# --no-chache-dir as we dont want to downloaded packages and their intermediate build files.
WORKDIR /
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir 
RUN pip install . --no-deps --no-cache-dir 

# "u" makes sure that any output from our script gets redirected to the terminal otherwise run docker logs
ENTRYPOINT ["python", "-u", "my_mlops_project_mnj/train_model.py"]