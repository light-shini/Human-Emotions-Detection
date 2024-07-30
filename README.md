
# Emotion Detection from Text

This project is a machine learning and NLP solution for detecting emotions from text. The model classifies text into various emotions such as sadness, joy, anger, love, and more.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data](#data)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project uses natural language processing (NLP) and machine learning techniques to classify emotions in text data. It employs various classifiers including Logistic Regression, Naive Bayes, Random Forest, and Support Vector Machine (SVM). The best-performing model is used for predictions.

## Project Structure

- `train.txt`: The dataset used for training and testing.
- `logistic_regression.pkl`: Trained Logistic Regression model.
- `label_encoder.pkl`: Label encoder for mapping emotions to numerical labels.
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer for text vectorization.
- `emotion_detection.ipynb`: Jupyter Notebook containing code for data processing, model training, and evaluation.
- `README.md`: Project documentation.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/emotion-detection.git
   cd emotion-detection
   ```

2. **Install the required packages:**
   Ensure you have Python 3.x installed. Then, run:
   ```bash
   pip install tensorflow==2.15.0
   pip install scikit-learn
   pip install pandas
   pip install numpy
   pip install seaborn
   pip install matplotlib
   pip install wordcloud
   pip install nltk
   ```

## Data

The dataset (`train.txt`) consists of text comments labeled with corresponding emotions. Each row contains a comment and its associated emotion.

## Data Preprocessing

Data preprocessing involves several steps:

1. **Text Cleaning:**
   - Remove special characters and non-alphabetic characters.
   - Convert text to lowercase.
   - Remove stopwords and perform stemming.

2. **Text Vectorization:**
   - TF-IDF (Term Frequency-Inverse Document Frequency) vectorization is used to convert text into numerical features.

3. **Label Encoding:**
   - Emotions are encoded into numerical values for model training.

Here is how the preprocessing is done:

```python
import nltk
import re
from nltk.stem import PorterStemmer

# Download NLTK stopwords
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)
```

## Model Training

The following steps outline the process of training the model:

1. **Train-Test Split:**
   - The dataset is split into training and testing sets.

2. **Vectorization:**
   - TF-IDF vectorization is used to transform text data into feature vectors.

3. **Model Training:**
   - Various classifiers are trained and evaluated, including Logistic Regression, Naive Bayes, Random Forest, and SVM.
   - The Logistic Regression model provided the best performance based on accuracy and classification metrics.

4. **Evaluation:**
   - Models are evaluated using accuracy scores and classification reports.

## Usage

To use the model for emotion prediction, follow these steps:

1. **Load the models and vectorizer:**
   ```python
   import pickle

   lg = pickle.load(open("logistic_regression.pkl", 'rb'))
   lb = pickle.load(open("label_encoder.pkl", 'rb'))
   tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", 'rb'))
   ```

2. **Predict Emotion:**
   ```python
   def predict_emotion(input_text):
       cleaned_text = clean_text(input_text)
       input_vectorized = tfidf_vectorizer.transform([cleaned_text])
       predicted_label = lg.predict(input_vectorized)[0]
       predicted_emotion = lb.inverse_transform([predicted_label])[0]
       return predicted_emotion

   sentence = "Your input text here"
   emotion = predict_emotion(sentence)
   print(f"Predicted Emotion: {emotion}")
   ```

## Results

The Logistic Regression model achieved the highest accuracy of approximately 83% on the test set. The classification report provides a detailed breakdown of the precision, recall, and f1-score for each emotion class.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
```

This README now includes a dedicated section for NLP, explaining how text data is cleaned and vectorized before being used for model training.
