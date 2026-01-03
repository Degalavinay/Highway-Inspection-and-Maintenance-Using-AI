# Highway-Inspection-and-Maintenance-Using-AI

Developed an AI-based system for highway inspection using deep learning and computer vision. Automated detection of potholes, cracks, and surface defects from drone/CCTV footage. Integrated GPS for accurate mapping and built a dashboard to prioritize maintenance, improving efficiency and reducing manual effort.

## Project Structure
```
Highway-Inspection-and-Maintenance-Using-AI/
├── App.py                  # Main Streamlit application
├── best.pt                 # Trained YOLO model weights
├── code_1.ipynb            # Jupyter notebook for initial code/experiments
├── Code_2.ipynb            # Jupyter notebook for model training/refinement
├── Data/                   # Directory containing video samples
├── Object Detection.v1i.coco/ # Dataset directory
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

## Requirements
The project requires Python and the following libraries:
- `streamlit`
- `opencv-python`
- `ultralytics`
- `torch`
- `numpy`

## Installation
1.  **Clone the repository** (or ensure you are in the project directory).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the main dashboard application:

```bash
streamlit run App.py
```

The application will open in your default web browser, allowing you to upload video footage or process existing samples for defect detection.
