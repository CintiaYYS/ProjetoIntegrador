FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install opencv-python pandas scikit-learn spacy
RUN python -m spacy download pt_core_news_lg

CMD ["python", "__init__.py"]
