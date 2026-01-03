# Highway Inspection and Maintenance Using AI

AI-powered highway inspection system using YOLOv8 for detecting highway elements like crash barriers, kerb paint, marking paint, and rubber speed breakers.

## Features

- **Object Detection**: Detects highway infrastructure elements using YOLOv8
- **Interactive UI**: Streamlit-based web interface for easy image upload and analysis
- **Multiple Models**: Includes pre-trained and fine-tuned YOLOv8 models
- **Detailed Results**: Provides annotated images, detection summaries, and confidence scores

## Project Structure

```
Highway-Inspection-and-Maintenance-Using-AI/
├── App.py                    # Main Streamlit application
├── Code.ipynb                # Training and analysis notebook
├── EDA.ipynb                 # Exploratory Data Analysis
├── yolov8m.pt               # Base YOLOv8 medium model
├── yolov8m_accuracy.pt      # Accuracy-optimized model
├── yolov8m_finetuned.pt     # Fine-tuned model (used by App.py)
├── requirements.txt          # Python dependencies
└── .gitignore               # Git ignore rules
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Highway-Inspection-and-Maintenance-Using-AI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run App.py
```

The app will open in your browser. Upload an image to detect highway elements.

## Models

- **yolov8m_finetuned.pt**: Primary model used in the application (fine-tuned on highway data)
- **yolov8m_accuracy.pt**: Optimized for accuracy
- **yolov8m.pt**: Base YOLOv8 medium model

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## License

This project is for educational and research purposes.
