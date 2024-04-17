# Use an appropriate base image
FROM python:3.8

# Set the working directory in the Docker container
WORKDIR /app

# Install any needed system dependencies here
RUN apt-get update && apt-get install -y libhdf5-dev

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of your application into the container at /app
COPY . .

# Run your application when the container launches
CMD ["python", "process_directory.py"]
