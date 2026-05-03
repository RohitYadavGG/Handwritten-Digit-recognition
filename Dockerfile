# Use a slim Python 3.9 image to keep the container small and fast
FROM python:3.9-slim

# Force Python to print logs immediately to the terminal (useful for debugging)
ENV PYTHONUNBUFFERED=1

# Hugging Face Spaces require running as a non-root user (User ID 1000)
# This prevents permission denied errors when your app tries to create folders like 'mobile_uploads'
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy the requirements file and install dependencies
COPY --chown=user ./requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt gunicorn

# Copy the rest of your app and set ownership to 'user'
COPY --chown=user . $HOME/app

# Start the Flask app using Gunicorn (Production-grade server)
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--workers", "2", "--threads", "4", "--timeout", "120", "app:app"]
