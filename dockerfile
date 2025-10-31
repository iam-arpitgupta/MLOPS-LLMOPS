FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN uv pip install -r requirements.txt

# Copy all project files
COPY . .

# Expose Streamlit’s default port
EXPOSE 8501

# Run only the Streamlit app
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
