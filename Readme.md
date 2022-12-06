# Social Unrest Prediction

This respository contains the models created to predict social unrest from news articles.

## NLP Models:
1. bi-LSTM Models
2. rnn Models
3. bi-rnn Models

## Breakdown of prediction:
1. News Classifier
2. Summarizer
3. Predictor

My interest in Socio-political stances motivated me to work on the infamous topic of social unrest, which is a significant concern across the globe, and getting the prediction insights could help minimize the loss and damage or even avoid the situation altogether. I collaborated with some like-minded friends for this project. We have tried to train multiple models to get the best predictor to predict the event. On the technical front, I used Python and libraries such as Pandas, Numpy, and, Scikit-learn. I performed Information extraction from given texts and classified them as actors, inter, the interaction between them, and date by using Tfidf Vectorizer, Named Entity Recognition, aggregated GloVe Embeddings, Transformer and BERT. We have also integrated the authentication of the news reported in ACLED for rumor verification. We have linked the model to verify and extract information from various data sources such as Twitter, Bing Search Engine, and the ACLED dataset. As the news can also be biased or politically influenced, we have performed the ground-check analysis to filter out such reports and sentiment analysis, followed by benchmarking and evaluation using the ACLED dataset. After training and validation scores, the better performing models were Stochastic gradient descent, Logistic Regression, and multinomial Naive Bayes classifiers.
