# Dockerfile

# 1. Start with the full Python base image instead of the slim one
FROM python:3.9

# 2. Update package lists and install required system libraries
# (It's good practice to keep these in, even on the full image)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# 3. Set the working directory inside the container
WORKDIR /code

# 4. Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# 5. Install the Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 6. Copy your application code into the container
COPY ./app.py /code/app.py

# 7. Expose the port Gradio will run on
EXPOSE 7860

# 8. The command to run when the container starts
CMD ["python", "app.py"]