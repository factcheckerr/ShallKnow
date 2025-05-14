# Use Python 3.10 slim image as the base
FROM python:3.10-slim-buster

# Install Miniconda
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda init && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Update PATH environment variable to include Conda
ENV PATH=/opt/conda/bin:$PATH

# Set the working directory
WORKDIR /app

# Copy the environment.yml to the working directory
COPY environment.yml .

# Create the Conda environment from environment.yml
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "triple_extraction_310", "/bin/bash", "-c"]

# Download the SpaCy model directly in the environment
RUN python -m spacy download xx_sent_ud_sm

# Install spaCy model
RUN python -m spacy download en_core_web_md

# Copy application code
COPY . .

# Expose the required port
EXPOSE 5000

# Set default command to run app.py in the Conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "triple_extraction_310", "python", "app.py"]

