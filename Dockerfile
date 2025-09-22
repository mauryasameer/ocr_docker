# Dockerfile

# 1. Start with the full Python base image
FROM python:3.9

# 2. Update package lists and install required system libraries
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# 3. Set the working directory inside the container
WORKDIR /code

# 4. Copy and install Python dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. --- START OF THE FIX ---
# Copy and run the pre-download script.
# This downloads the models and saves them as a layer in the Docker image.
COPY ./download_models.py /code/download_models.py
RUN python download_models.py
# --- END OF THE FIX ---

# 6. Copy your application code into the container
COPY ./app.py /code/app.py

# 7. Expose the port Gradio will run on
EXPOSE 7860

# 8. The command to run when the container starts
CMD ["python", "app.py"]