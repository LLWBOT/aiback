# Use a Python 3.11 base image for better performance and security
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Set the FLASK_APP environment variable so Flask knows which file to run
ENV FLASK_APP=app.py

# Expose port 5000 for the Flask application
EXPOSE 5000

# Run the application with Flask's built-in development server
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
