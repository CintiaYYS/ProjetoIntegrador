FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    wget \
    curl \
    ffmpeg \
    chromium-driver \
    chromium \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --upgrade pip

RUN pip install \
    pandas \
    scikit-learn \
    spacy \
    swifter \
    opencv-python \
    Pillow \
    scikit-image \
    ultralytics \
    pytesseract \
    transformers \
    torch torchvision torchaudio \
    playwright \
    requests

RUN playwright install --with-deps

RUN python -m spacy download pt_core_news_lg

RUN mkdir -p dados/imagens dados/modelos dados/processados


CMD ["python", "__init__.py"]