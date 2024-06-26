# Use an official Python runtime as a parent image
FROM python:3.10.9-slim

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Specify the command to run Python script (adjust as necessary)
CMD ["sh", "-c", "python3 script.py"]
