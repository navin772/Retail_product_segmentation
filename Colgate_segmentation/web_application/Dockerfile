# Use Python as the base image
FROM python:3.11-slim

# Install Git
RUN apt-get update && apt-get install -y git

# Set the working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/ultralytics/ultralytics

# Change directory to ultralytics
WORKDIR /app/ultralytics

# Modify requirements.txt file
RUN sed -i 's/opencv-python>=4.6.0/opencv-python-headless/' requirements.txt

# Install the requirements
RUN pip install -e .

# Go back to the original working directory
WORKDIR /app

# Copy the contents of the current directory to the Docker container
COPY . /app

EXPOSE 8501

RUN pip install -r requirements.txt
# (Optional) Add any additional commands or configurations here if needed
ENTRYPOINT ["streamlit", "run"]
# Define the default command to run when the container starts
CMD ["main.py"]
