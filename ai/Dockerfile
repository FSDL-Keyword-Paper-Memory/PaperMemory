FROM python:3.8.13-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -c "import nltk; nltk.download('stopwords')"
COPY src ./src
COPY run ./run
COPY logs ./logs
ENV PYTHONPATH .
ENV FLASK_RUN_HOST 0.0.0.0
ENV MODEL all-MiniLM-L6-v1
ENV THRESHOLD 0.53
CMD flask --app run/app.py run
