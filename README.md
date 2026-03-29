# spam_detection
# SMS Spam Detection

This project is a simple machine learning model that classifies SMS messages as either spam or not spam (ham).

## About the Project
Nowadays, spam messages are very common and can be annoying or even harmful. The aim of this project is to build a basic spam detection system using machine learning techniques.

## Technologies Used
- Python
- Numpy
- Pandas
- NLTK (for text processing)
- Scikit-learn

## Dataset
The dataset used in this project is the SMS Spam Collection Dataset from Kaggle.

## How It Works
1. The dataset is loaded and cleaned.
2. Text preprocessing is done:
   - Convert text to lowercase
   - Remove punctuation and numbers
   - Remove stopwords
   - Perform lemmatization
3. Text is converted into numerical form using TF-IDF.
4. The dataset is split into training and testing data.
5. A Naive Bayes model is trained.
6. The model predicts whether a message is spam or not.

## How to Run
1. Install required libraries:
   pip install numpy pandas nltk scikit-learn kagglehub

2. Run the Python file or notebook.

3. The program will output the accuracy of the model.

## Output
The model prints the accuracy score based on test data.

## Conclusion
This project shows how basic NLP and machine learning can be used to detect spam messages effectively.
