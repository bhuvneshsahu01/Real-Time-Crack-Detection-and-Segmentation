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

- **Number of Images**: [Specify number]
- **Types of Cracks**: [Briefly describe types]
- **Annotation Tool**: LabelMe

## Models Used

1. **YOLOv8**: Used for object detection to identify crack regions.
2. **Detectron2**: Utilized for instance segmentation to segment cracks in the images.
3. **Segment Anything Model (SAM)**: Used to provide additional segmentation capabilities, enhancing the detection accuracy.

## Training and Fine-Tuning

Each model was fine-tuned on the custom dataset to optimize performance for crack detection and segmentation. Details on how the models were trained:

- **YOLOv8**: [Training details, epochs, learning rate, etc.]
- **Detectron2**: [Training details, epochs, learning rate, etc.]
- **SAM**: [Training details, epochs, learning rate, etc.]

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/crack-detection-segmentation.git
    cd crack-detection-segmentation
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the pre-trained weights and place them in the `models/` directory. [Provide download link if available]

## Usage

To run the real-time crack detection and segmentation:

1. Open the terminal and navigate to the project directory.
2. Run the following command:
    ```bash
    python run_detection.py --source path_to_input_data
    ```

- `--source`: Path to the input data (image or video) for detection.

## Results

The models were evaluated on a separate validation set, achieving the following metrics:

- **YOLOv8**: [Specify mAP, Precision, Recall, etc.]
- **Detectron2**: [Specify mAP, Precision, Recall, etc.]
- **SAM**: [Specify mAP, Precision, Recall, etc.]

Sample results are provided in the `results/` directory.

## Streamlit Application

A Streamlit application was developed to demonstrate the real-time crack detection and segmentation capabilities. To run the Streamlit app:

```bash
streamlit run app.py
