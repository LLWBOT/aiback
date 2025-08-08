# Use an official Python runtime as a parent image
# We are upgrading the version to 3.10 to support spellchecker==0.7.1
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download necessary NLTK data during the build process
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# Copy the rest of the application files into the container
COPY . .

# Expose port 5000 for the Flask application
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
