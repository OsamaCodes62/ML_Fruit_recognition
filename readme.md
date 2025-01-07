# Fruit Classification Using Convolutional Neural Networks (CNN)

This project leverages a deep learning model to classify images of fruits into multiple categories using a Convolutional Neural Network (CNN). The model is trained on a labeled dataset of fruit images and tested for accuracy and performance.

---

## Features
- Data augmentation and preprocessing for robust training using **ImageDataGenerator**.
- Multiclass classification using a CNN built with **TensorFlow/Keras**.
- Visualizations of training accuracy and validation accuracy over epochs.
- Confusion matrix for detailed performance analysis.
- Saved trained model for reuse or deployment.

---

## Dataset

The dataset used for this project contains a variety of fruit images organized into categories. It can be downloaded from Kaggle:

[**Fruits Dataset on Kaggle**](https://www.kaggle.com/datasets/moltean/fruits/data)

### Example Dataset Structure:
```
Dataset/
│
├── Training/
│   ├── Apple/
│   ├── Banana/
│   ├── Orange/
│   └── (Other fruit categories)
│
└── Test/
    ├── Apple/
    ├── Banana/
    ├── Orange/
    └── (Other fruit categories)
```

---

## Requirements

Install the required dependencies using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## Getting Started

1. **Download the Dataset**:
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/moltean/fruits/data).
   - Extract the dataset and organize it into `Training` and `Test` directories as shown above.

2. **Training the Model**:
   - Open the `training.ipynb` notebook.
   - Follow the steps to preprocess the data, define the CNN architecture, train the model, and save it as `trained_model.h5`.

3. **Testing the Model**:
   - Open the `testing.ipynb` notebook.
   - Load the trained model, preprocess the test data, and evaluate its performance.
   - Generate a confusion matrix and calculate the accuracy score.

---

## Results

### Accuracy Visualization
The training process generates visualizations to show how accuracy improves over epochs:
- **Training Accuracy vs. Epochs**
- **Validation Accuracy vs. Epochs**

### Confusion Matrix
A detailed confusion matrix is plotted to visualize how well the model classifies each fruit category.

---

## Directory Structure
```
.
├── Training/                  # Training dataset
├── Test/                      # Test dataset
├── training.ipynb             # Jupyter Notebook for training the model
├── testing.ipynb              # Jupyter Notebook for testing the model
├── trained_model.h5           # Saved trained model
├── requirements.txt           # File with project dependencies
├── README.md                  # Project documentation
```

---

## Dataset properties ##

Total number of images: 94110.

Training set size: 70491 images (one fruit or vegetable per image).

Test set size: 23619 images (one fruit or vegetable per image).

Number of classes: 141 (fruits, vegetables and nuts).

Image size: 100x100 pixels.

Filename format: 

image_index_100.jpg (e.g. 32_100.jpg) or 

r_image_index_100.jpg (e.g. r_32_100.jpg) or 

r?_image_index_100.jpg  (e.g. r2_32_100.jpg)

where "r" stands for rotated fruit. "r2" means that the fruit was rotated around the 3rd axis. 
"100" comes from image size (100x100 pixels).

Different varieties of the same fruit (apple for instance) are stored as belonging to different classes.

## Repository structure ##

Folders [Training](Training) and [Test](Test) contain images for training and testing purposes.

## Alternate download ##

The Fruits-360 dataset can be downloaded from: 

**Kaggle** [https://www.kaggle.com/moltean/fruits](https://www.kaggle.com/moltean/fruits)

**GitHub** [https://github.com/fruits-360](https://github.com/fruits-360)

## How to cite ##

Mihai Oltean, __Fruits-360 dataset__, 2017-.

---

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/fruit-classification-cnn.git
   ```
2. Navigate to the project directory:
   ```bash
   cd fruit-classification-cnn
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open the Jupyter Notebooks:
   ```bash
   jupyter notebook
   ```
   - Run `training.ipynb` to train the model.
   - Run `testing.ipynb` to test the model and analyze results.

---

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

---

## Contributions
Contributions are welcome! Feel free to fork this repository, create an issue, or submit a pull request.

---

If you'd like to include specific content for `requirements.txt` or further tweaks, let me know!
