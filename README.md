# Real-Time Crack Detection and Segmentation

This project focuses on real-time crack detection and segmentation using state-of-the-art deep learning models. It was developed as part of a summer internship, with all data collected and annotated specifically for this task.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Training and Fine-Tuning](#training-and-fine-tuning)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Streamlit Application](#streamlit-application)
- [Docker Image](#docker-image)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to detect and segment cracks in various structures using deep learning techniques. By leveraging models like YOLOv8, Detectron2, and Segment Anything Model (SAM), the project achieves high accuracy and real-time performance in identifying and localizing cracks. The models were fine-tuned on a custom dataset, ensuring robustness and adaptability to different types of cracks.

## Dataset

The dataset used for this project was collected manually and includes various types of cracks, providing a diverse range of examples. Ground truth annotations were created using the LabelMe tool, which helped in generating accurate labels for training and evaluation.

- **Number of Images**: 430.
- **Image dimensions**: 480*480.
- **Types of Cracks**:  various types of (deep, wide, thin, hairline) on various surfaces (walls, roofs, concrete, plaster)
- **Annotation Tool**: LabelMe

## Models Used

1. **YOLOv8**: by Ultralytics.
2. **Detectron2**: by Meta.
3. **Segment Anything Model (SAM)**: By Meta.

## Training and Fine-Tuning

Each model was fine-tuned on the custom dataset to optimize performance for crack detection and segmentation. Details on how the models were trained:

** All models are trained with 200 epochs **

## Usage

To run the real-time crack detection and segmentation:

1. Open the terminal and navigate to the project directory.
2. Run the following command:
    ```bash
    python run_detection.py --source path_to_input_data
    ```

- `--source`: Path to the input data (image or video) for detection.


Sample results are provided in the `results/` directory.

## Streamlit Application

A Streamlit application was developed to demonstrate the real-time crack detection and segmentation capabilities. To run the Streamlit app:

```bash
streamlit run app.py
