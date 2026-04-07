# Dockerfile

# 1. Start with the full Python base image
FROM python:3.13

# 2. Add non-root user (Required for Hugging Face Spaces)
RUN useradd -m -u 1000 user

# 3. Update package lists and install required system libraries
# Default: only PaddleOCR requirements
# Use BUILD_ENGINES="paddle,easyocr,tesseract" to install others
ARG BUILD_ENGINES="paddle"

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    $(if echo "$BUILD_ENGINES" | grep -q "tesseract"; then echo "tesseract-ocr"; fi) \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory inside the container
WORKDIR /code

# 5. Copy and install Python dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Install optional engines based on BUILD_ENGINES
RUN if echo "$BUILD_ENGINES" | grep -q "easyocr"; then \
        pip install --no-cache-dir easyocr; \
    fi && \
    if echo "$BUILD_ENGINES" | grep -q "tesseract"; then \
        pip install --no-cache-dir pytesseract; \
    fi

# 6. Change ownership of the working directory to the new user
RUN chown -R user:user /code

# 7. Switch to the non-root user
USER user

# 8. Set HOME environment variable so models download to a writable directory
ENV HOME=/code


# 10. Copy your application code into the container
COPY --chown=user:user ./src /code/src
COPY --chown=user:user ./data /code/data
COPY --chown=user:user ./app.py /code/app.py

# 11. Expose the port Gradio will run on
EXPOSE 7860

# 12. Use the provided command to run the app
CMD ["python", "app.py"]