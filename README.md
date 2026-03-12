# PaddleOCR App with Gradio

This project is a web-based Optical Character Recognition (OCR) application powered by [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) and [Gradio](https://gradio.app/). It allows you to upload an image and automatically detects, extracts, and highlights text using bounding boxes, providing confidence scores for each detection.

## Features
- **Accurate Text Extraction:** Leverages PaddleOCR for robust text detection and recognition.
- **Visual Bounding Boxes:** Draws precise bounding boxes around detected text in the uploaded image.
- **Interactive UI:** Built with Gradio for a seamless and user-friendly web interface.
- **Containerized:** Includes a `Dockerfile` for consistent, reproducible deployments.

---

## 🚀 How to Deploy for Free (Like GitHub Pages)

Because this application relies on a Python backend and deep learning models (PaddleOCR), it **cannot** be hosted on GitHub Pages (which only supports static HTML/CSS). 

However, you can easily deploy it for free on **Hugging Face Spaces** or **Render**, which are the industry standards for showcasing ML/Python portfolios.

### Option 1: Deploy on Hugging Face Spaces (Recommended)
Hugging Face Spaces offers free hosting for ML apps and natively supports Docker and Gradio.

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and create a new Space.
2. Enter a name for your Space.
3. Choose **Docker** as the Space SDK (since this project uses a custom `Dockerfile` to install system libraries).
4. Choose **Public** for visibility.
5. Clone the repository Hugging Face provides, copy your project files into it, and push it back:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   cp -r /path/to/this/project/* YOUR_SPACE_NAME/
   cd YOUR_SPACE_NAME
   git add .
   git commit -m "Initial commit"
   git push
   ```
6. Hugging Face will automatically build your Docker container and launch the app. You can now share the link in your portfolio!

### Option 2: Deploy on Render
Render provides a free tier for Docker-based web services.

1. Push this project code to your own GitHub repository.
2. Create an account on [Render.com](https://render.com/).
3. Create a **New Web Service** and connect your GitHub repository.
4. Render will detect the `Dockerfile` and automatically build and deploy your app.

---

## 💻 How to Run Locally

### Using Docker (Simplest way)
If you have Docker installed, you can build and run the app without manually configuring Python environments:

```bash
# Build the Docker image
docker build -t paddleocr-app .

# Run the container
docker run -p 7860:7860 paddleocr-app
```
Then, open your browser and navigate to `http://localhost:7860`.

### Using Python setup
If you prefer running it directly via Python (requires Python 3.9+):

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```
