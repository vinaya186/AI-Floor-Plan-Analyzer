# 🏠 AI Floor Plan Analyzer

An AI-powered application that analyzes architectural floor plans using computer vision and generates object quantity reports and preliminary cost estimates.

## 📌 Project Overview

The AI Floor Plan Analyzer is a Python-based application developed using Streamlit and YOLOv8. It detects objects in architectural floor plans, counts the detected objects, performs quantity analysis, and generates downloadable Excel reports.

The application supports both image files and multi-page PDF floor plans.

## ✨ Key Features

- **Floor Plan Upload:** Upload PNG, JPG, and PDF files.
- **Object Detection:** Detect architectural objects using a YOLOv8 model.
- **Object Counting:** Automatically count detected objects.
- **Quantity Analysis:** Generate page-wise and overall object quantity reports.
- **Cost Estimation:** Calculate preliminary estimated costs using user-provided unit rates.
- **Excel Report Generation:** Generate a downloadable Excel workbook containing five sheets:
  1. Overall Summary
  2. Page-wise Report
  3. Detailed Detection
  4. Cost Estimate
  5. Project Summary

## 🛠️ Technologies Used

- Python
- Streamlit
- YOLOv8
- Ultralytics
- PyTorch
- Pandas
- Pillow
- PyMuPDF
- OpenPyXL
- Hugging Face Hub

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/vinaya186/AI-Floor-Plan-Analyzer.git
```

### 2. Navigate to the project folder

```bash
cd AI-Floor-Plan-Analyzer
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🚀 How It Works

1. Upload an architectural floor plan.
2. Select the required PDF pages if applicable.
3. Run the AI-based object detection.
4. View the annotated floor plan and detected objects.
5. Review object counts and quantity reports.
6. Enter unit rates to calculate preliminary cost estimates.
7. Download the generated Excel report.

## 🧠 AI Model

The application uses a pretrained YOLOv8 object detection model hosted on Hugging Face:

`mudasir13cs/floorcad-yolov8n-detect`

The model weights are downloaded automatically when required.

## 🎯 Project Objective

The objective of this project is to explore how artificial intelligence and computer vision can assist in analyzing architectural floor plans and simplify object identification, quantity reporting, and preliminary cost estimation.

## ⚠️ Note

The application provides AI-based object detections and preliminary estimates. Results should be reviewed and verified by qualified professionals before being used for actual construction, budgeting, or engineering decisions.


**Vinaya Khaire**

B.Tech – Final Year Project
