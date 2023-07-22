# Use Ubuntu minimal as the base image
FROM ubuntu:latest

# Set environment variables to avoid any interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary dependencies for your application
RUN apt-get update && apt-get install -y python3 python3-pip libgl1 libsm6 libxext6

# Set the locale to avoid potential encoding issues
RUN apt-get install -y locales && locale-gen en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Copy your application code
COPY . /app

# Install streamlit and other requirements
RUN pip3 install --upgrade pip && pip3 install -r /app/requirements.txt

# Expose the port for Streamlit
EXPOSE 8501

# Set the working directory
WORKDIR /app

# Run your Streamlit app
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]
