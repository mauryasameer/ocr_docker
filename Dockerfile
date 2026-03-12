# Dockerfile

# 1. Start with the full Python base image
FROM python:3.9

# 2. Add non-root user (Required for Hugging Face Spaces)
RUN useradd -m -u 1000 user

# 3. Update package lists and install required system libraries
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory inside the container
WORKDIR /code

# 5. Copy and install Python dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 6. Change ownership of the working directory to the new user
RUN chown -R user:user /code

# 7. Switch to the non-root user
USER user

# 8. Set HOME environment variable so models download to a writable directory
ENV HOME=/code

# 9. Copy and run the pre-download script.
# This downloads the models and saves them as a layer in the Docker image.
COPY --chown=user:user ./download_models.py /code/download_models.py
RUN python download_models.py

# 10. Copy your application code into the container
COPY --chown=user:user ./app.py /code/app.py

# 11. Expose the port Gradio will run on
EXPOSE 7860

# 12. The command to run when the container starts
CMD ["python", "app.py"]