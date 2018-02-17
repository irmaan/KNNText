## Text Classification Using K-NN

## Data :

Combined_News_DJIA is a dataset that contains daily news headlines and the corresponding stock market movement of the Dow Jones Industrial Average (DJIA) from 2008 to 20166. The dataset was created by Aaron Sun and is available on Kaggle6. The dataset can be used for various tasks such as sentiment analysis, natural language processing, and stock market prediction.

Some examples of the headlines in the dataset are:

b"Georgia 'downs two Russian warplanes' as countries move to brink of war"
b'Why wont America and Nato help us? If they wont help us now, why did we help them in Iraq?'
b'Remember that adorable 9-year-old who sang at the opening ceremonies? That was fake, too.'
b'North Korea says US within missile range'
b'Nestle buys 60% of Chinese candy maker Hsu Fu Chi for $1.7 billion'
The dataset also contains a binary label for each date, indicating whether the DJIA increased or decreased on that day. The label is 1 if the DJIA increased, and 0 if it decreased.

 ## Workflow:
 In this program, a group of  25 related news text are considered as a document. First, Bigrams and Trigrams are extracted from the training data, then their frequency is obtained. In the next step, according to the frequency of Bigram and Trigrams, the Idf vector is obtained. Also, the TF table is made from the number of repetitions of words and then it is normalized according to the total number of repetitions of words (the same as bi and trigram). The tf-idf table is also obtained from the product of the elements of the tf table in the corresponding Idf values of each word (bi and tri) and both the TF and TF_IDF tables are stored as csv.
In order to determine the degree of similarity of each test data (document containing 25 news texts) with training data, pair by pair comparisons should be done. In this way, bigram and trigram should be extracted for each test data, and its TF and IDF vectors should be produced and its TF_IDF value should be obtained. The storage of the TF vector is done in a dense form due to performing many calculations in a matrix with more than 450,000 columns (for only 30% of the training data) and also because it is sparse, that is, only the words that were in the vocabulary in the form of a dictionary word and its number are stored for each document. In this method, by using the dictionary keys stored for the TF vector of each of the training and test documents and multiplying their TF-IDF values together and then dividing by the product of their length (i.e. cosine relationship formula), we can determine the degree of similarity between two data. 
Each test data is compared with all the training data and their cosine similarity is extracted, and finally according to K in the KNN algorithm, the K closest documents are considered in terms of similarity, and according to the dominant class (i.e. the number of documents belonging to one class is more than another class) the classification is done. Below is the execution result for 5 values of K on the training and test data.


## Results:

- K = 10	Train = 81.89	Test = 79.23
- K = 20	Train = 82.41	Test = 80.34
- K = 30	Train = 85.71	Test = 83.23
- K = 40	Train = 88.61	Test = 86.78
- K = 60	Train = 81.54	Test = 79.93

