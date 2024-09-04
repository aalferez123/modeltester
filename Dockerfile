# Python image to use.
FROM python:3.12.5-bullseye

# Set the working directory to /app
WORKDIR /modeltestercr

# copy the requirements file used for dependencies
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip3 install -r requirements.txt

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Run app.py when the container launches
ENTRYPOINT ["python3", "app.py"]
