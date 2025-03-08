# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt --trusted-host files.pythonhosted.org --trusted-host pypi.org --trusted-host pypi.python.org
RUN pip install -e .

# Define environment variable
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Run the script by default when the container starts
ENTRYPOINT ["connectomix"]
