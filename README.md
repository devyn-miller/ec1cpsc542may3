---
runme:
  id: 01HSEEWGP8NWV3ZH52D49ZNGHM
  version: v3
---

# CPSC542 Assignment 2

## Student Information

- **Student Name:** Devyn Miller
- **Student ID:** 2409539

## Collaboration

- **In-class collaborators:** Hayden Fargo (not working on the same code, but just bouncing ideas off of one another)

## Resources

- **Resources used:**
   - [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/%7Evgg/data/pets/)
   - [oxford_iiit_pet](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet)
   - Copilot
   - Perplexity

## Data Source

- __Data source:__ [oxford_iiit_pet](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet) `dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)`

## Code Repository

- **GitHub Repository:** [Assignment 2](https://github.com/devyn-miller/the-final-assignment2-cpsc542.git)

## **Project Organization and Pipeline Overview**

This project encapsulates my journey in developing an image segmentation pipeline that leverages deep learning and computer vision techniques. I developed a comprehensive pipeline for the segmentation of images from the Oxford-IIIT Pet Dataset, focusing on accurately distinguishing pets from their backgrounds across various settings and poses. This project harnesses the power of TensorFlow and Keras libraries to achieve its goals. Through careful organization and documentation, I aimed to create a transparent and reproducible workflow that addresses the challenges of pet image segmentation. 

### Data Preprocessing

I started by preprocessing the data to make it suitable for training a deep learning model. This involved loading the dataset using TensorFlow Datasets (TFDS), normalizing the pixel values of the images to the range [0, 1], and resizing both the images and their corresponding segmentation masks to a uniform dimension of 128x128 pixels. The preprocessing steps are encapsulated in the `preprocess` function within `src/preprocessing.py`.


### Data Augmentation

To improve the model's generalization capability, I implemented data augmentation techniques, including random horizontal flipping of the images and masks. This augmentation is performed on-the-fly during training to introduce variability in the training data without increasing its size. The augmentation logic is defined in the `augment` function in `src/augmentation.py`.


### Model Architecture

For the segmentation task, I utilized a U-Net architecture, known for its effectiveness in image segmentation tasks. The U-Net model comprises a pretrained MobileNetV2 as the encoder and a series of upsample blocks as the decoder, facilitating precise localization. The model architecture is defined in `src/model.py`.


### Training

The model is compiled and trained using the Adam optimizer and Sparse Categorical Crossentropy loss function, suitable for multi-class segmentation tasks. Training involves feeding the preprocessed and augmented images to the model, with early stopping implemented to prevent overfitting. The training process is managed by the `train_model` function in `src/training.py`.


### Evaluation and Visualization

Post-training, the model's performance is evaluated using metrics such as accuracy, precision, recall, F1 score, and Intersection over Union (IoU). Additionally, I implemented Grad-CAM visualizations to interpret the model's predictions, highlighting the regions of the image that contributed most to the segmentation decision. The evaluation and visualization steps are detailed in `src/metrics.py` and `src/grad_cam.py`.


### Conclusion

This project demonstrates a structured approach to solving an image segmentation problem using deep learning. By carefully preprocessing the data, augmenting it to enhance model robustness, and employing a powerful U-Net architecture, I was able to achieve precise segmentation of pets from their backgrounds. The project's modular design ensures each component is easily understandable and modifiable for future enhancements or adaptations to similar tasks.

# Project Structure

Below is the tree structure of the project repository, detailing the organization and contents of each file and directory:

the-final-assignment2-cpsc542

├── README.md                   # Project overview, setup instructions, and additional notes

├── src                         # Source code for the project

│   ├── preprocessing.py        # Contains functions for data loading and preprocessing

│   ├── augmentation.py         # Implements data augmentation techniques

│   ├── model.py                # Defines the U-Net model architecture

│   ├── training.py             # Manages the model training process

│   ├── metrics.py              # Evaluation metrics and performance analysis

│   └── grad_cam.py             # Grad-CAM visualizations for model interpretation

├── figures                     # Directory for storing static figures and plots

│   └── ...                     # Various PNG images from initial exploratory data analysis

│   └── main.ipynb              # Main notebook with project walkthrough, including EDA and results

└── requirements.txt            # Lists the project's Python dependencies for replication


Each component of the project is modularized, allowing for easy understanding and modification. The `src` directory contains the core logic for preprocessing, augmentation, model definition, training, and evaluation. Static figures generated during the project are stored in the `figures` directory. The `requirements.txt` file lists all the necessary Python packages to ensure the project can be replicated in different environments.


- `README.md`: Project overview, setup instructions, and additional notes.
- `src`: Source code for the project.
  - `preprocessing.py`: Contains functions for data loading and preprocessing.
  - `augmentation.py`: Implements data augmentation techniques.
  - `model.py`: Defines the U-Net model architecture.
  - `training.py`: Manages the model training process.
  - `metrics.py`: Evaluation metrics and performance analysis.
  - `grad_cam.py`: Grad-CAM visualizations for model interpretation.
- `figures`: Directory for storing static figures and plots.
  - Various PNG images from initial exploratory data analysis.
  - `model_structures`: PNG Images for various model architectures I experimented with.
- `main.ipynb`: Main notebook with project walkthrough, including EDA and results.
- `model_history.json`: Stores the training history of the model for analysis.
- `models`: Due to size constraints, the trained model is hosted externally. A link is provided in the repository for access.

# Additional notes below — please read.

1. The u-net model was too large to push to github and can be found at the following link: https://drive.google.com/file/d/1zoP3UmzBmihgdkKlurVVzK1fd9ylbaqz/view?usp=sharing
   ![Alt text](model.png?raw=true)
2. All of the plots and figures (EDA, grad-CAM, etc) for this project can be found in the Jupyter notebook entitled `main.ipynb`. The report only contains the metrics table and graphs to show model performance.
3. As described in my writeup, I modularized things from an initial Jupyter notebook that contained everything. The png images saved in the `/figures` folder are from that inital ipynb, while those in `main.ipynb` were generated when running the modularized project.
