# Spam or Ham Email Detection Using TensorFlow

## Overview
This project implements a **spam email classifier** using **deep learning** in Python. The system uses **LSTM (Long Short-Term Memory)** networks to capture patterns in email text and predict whether a message is **spam** or **non-spam (ham)**.

---

## Dataset
- Contains two columns: `text` (email content) and `label` (`spam` or `ham`).  
- Original dataset is **imbalanced**, with more non-spam emails than spam.  
- The majority class (ham) is **downsampled** to balance the dataset and prevent bias.

---

## Data Preprocessing
- **Text cleaning:** Removes unnecessary headers like "Subject", punctuation, and stopwords (common words like “the”, “and”, “is”).  
- **Reason:** Reduces noise, shrinks vocabulary, and helps the model focus on meaningful words.  
- **Exploratory analysis:** Word clouds and class distribution plots help understand frequent words in spam and ham emails.

---

## Text Vectorization
- **Tokenization:** Converts each word into a unique integer to prepare for modeling.  
- **Padding:** Ensures all sequences have a fixed length so the neural network can process them uniformly.  
- **Label encoding:** Converts text labels to binary (`spam = 1`, `ham = 0`) for classification.

---

## Model Architecture
- **Embedding Layer:** Learns dense vector representations of words.  
- **LSTM Layer:** Captures sequential patterns and context in emails.  
- **Dense Layer:** Extracts relevant features.  
- **Output Layer:** Sigmoid activation predicts probability of spam.

**Why LSTM?**  
- Emails are sequences of words, and word order matters. LSTM can remember important sequences across long text inputs.

---

## Training Strategy
- **Loss Function:** Binary cross-entropy for binary classification.  
- **Optimizer:** Adam for efficient gradient descent.  
- **Callbacks:**  
  - EarlyStopping stops training if validation accuracy does not improve.  
  - ReduceLROnPlateau lowers learning rate when validation loss plateaus.  

- Training uses **train/test split**, batch processing, and validation for reliable performance assessment.

---

## Evaluation
- The model is evaluated on the test set using **accuracy and loss metrics**.  
- The training and validation performance is monitored over epochs to detect overfitting.

---

## Tools & Libraries Used
- **Python**  
- **Pandas & NumPy:** Data handling  
- **Matplotlib & Seaborn:** Visualization  
- **NLTK:** Text preprocessing (stopwords)  
- **WordCloud:** Text visualization  
- **TensorFlow / Keras:** Deep learning model (Embedding + LSTM)  
- **Scikit-learn:** Train/test split and preprocessing utilities  

---

## Key Notes
- Preprocessing is crucial to reduce noise and focus on meaningful words.  
- LSTM is effective for sequential text data like emails.  
- The approach balances dataset classes to improve model fairness.  
- The workflow is end-to-end: **raw email → preprocessing → tokenization → padded sequences → LSTM model → spam prediction**.  

---


## Explanation of Steps

1. **Raw Emails** – Original messages with all text and headers.  
2. **Text Cleaning** – Remove "Subject", punctuation, and stopwords to reduce noise.  
3. **Tokenization** – Convert words into unique integer indices.  
4. **Padding/Truncation** – Standardize sequence length for the neural network.  
5. **Embedding Layer** – Learns dense vector representations for each word.  
6. **LSTM Layer** – Captures word order and sequential patterns in the text.  
7. **Dense Layer** – Extracts higher-level features from the LSTM output.  
8. **Output Layer** – Sigmoid neuron outputs probability of spam.  
