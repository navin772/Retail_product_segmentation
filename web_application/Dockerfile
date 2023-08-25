# Use Python as the base image
FROM registry.suse.com/bci/python:3.10

# Install Git -> git-core present by default in suse BCI
RUN zypper refresh && \
    zypper clean --all

# Set the working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/ultralytics/ultralytics

# Change directory to ultralytics
WORKDIR /app/ultralytics

# Delete the requirements.txt file
RUN rm requirements.txt

# Create a new requirements.txt file for ultralytics
RUN echo "opencv-python-headless" >> requirements.txt && \
    echo "matplotlib>=3.2.2" >> requirements.txt && \
    echo "pyyaml>=5.3.1" >> requirements.txt && \
    echo "requests>=2.23.0" >> requirements.txt && \
    echo "scipy>=1.4.1" >> requirements.txt && \
    echo "tqdm>=4.64.0" >> requirements.txt && \
    echo "pandas>=1.1.4" >> requirements.txt && \
    echo "psutil" >> requirements.txt && \
    echo "py-cpuinfo" >> requirements.txt

# Install the requirements
RUN pip install --no-cache -e .

# Go back to the original working directory
WORKDIR /app

# RUN rm -r ultralytics

# Copy the contents of the current directory to the Docker container
COPY . /app

EXPOSE 8501

# Install the requirements
RUN pip install --no-cache -r requirements.txt

ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]