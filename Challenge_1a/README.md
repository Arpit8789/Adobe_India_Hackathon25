# Adobe India Hackathon 2025 – Challenge 1a

## Overview
This solution extracts structured data from PDF documents and outputs corresponding JSON files.

## Directory Structure

.
├── process_pdfs.py
├── heading_classifier.pkl
├── label_encoder.pkl
├── Dockerfile
├── requirements.txt
└── sample_dataset/
    ├── pdfs/
    ├── outputs/
    └── schema/


## Build Instructions
bash
docker build --platform linux/amd64 -t pdf-processor .


## Run Instructions
bash
docker run --rm   -v $(pwd)/sample_dataset/pdfs:/app/input:ro   -v $(pwd)/sample_dataset/outputs:/app/output   --network none   pdf-processor


## Notes
- Works offline with no internet access.
- Efficient memory and CPU usage.
- Model size under 200MB.
