# Email Spam Classifier

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Overview
The Email Spam Classifier is a machine learning project that classifies emails as spam or not spam. This project leverages natural language processing (NLP) techniques and machine learning algorithms to detect spam emails with high accuracy.

## Installation
To get a local copy up and running, follow these steps:

1. **Clone the repository**
   ```
   git clone https://github.com/Amartya-007/Email-Sms-Classifir.git
   cd Amartya-007---Email-Sms-Classifir
   ```
Create and activate a virtual environment (optional but recommended)
```
python -m venv venv
source venv/bin/activate
```
On Windows, use 
```
venv\Scripts\activate
```
Install the required packages
```
pip install -r requirements.txt
```
## Usage
1. **Run the Jupyter notebook**
   - Open `email_spam_classifier.ipynb` in Jupyter Notebook or Jupyter Lab to see the preprocessing, training, and evaluation steps.

2. **Run the Streamlit app**
   - To use the interactive Streamlit application, execute the following command:
     ```sh
     streamlit run streamlit_app.py
     ```
   - This will start a local server, and you can use your browser to interact with the email spam classifier.

## Dataset
- The dataset (`spam.csv`) consists of labeled email data used for training and testing the model. It should be placed in the same directory as the other files before running any scripts.

## Model
- The project employs a machine learning model to classify emails. The model and the text vectorizer are saved as `model.pkl` and `vectorizer.pkl`, respectively. These files are loaded in the Streamlit app for making predictions.

## Results
- The performance of the model is evaluated using various metrics such as accuracy, precision, recall, and F1-score. Detailed evaluation results can be found in the `email_spam_classifier.ipynb` notebook.


## Acknowledgements
- [NLTK](https://www.nltk.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Streamlit](https://streamlit.io/)
