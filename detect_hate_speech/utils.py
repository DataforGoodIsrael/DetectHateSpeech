import nltk
import pandas as pd
import re
import string
import wordninja
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report


nltk.download('stopwords')

STOPWORDS_NLTK = nltk.corpus.stopwords.words('english') + ['rt']
MIN_LEN_WORD = 2


class DataPreProcessing:
    """
    - Remove url, mentions, hashtags (and try to split the word), punctuations
    - Tokenization - Converting a sentence into list of words
    - Remove stopwords
    - Lammetization/stemming - Tranforming any form of a word to its root word
    """

    def __init__(self, tweets_column):
        self.tweets_column = tweets_column

    @staticmethod
    def get_urls(sentence):
        return " ".join([word for word in sentence.split()
                         if word.lower().startswith("http")])

    @staticmethod
    def remove_urls(sentence):
        return " ".join([word for word in sentence.split()
                         if not word.lower().startswith("http")])

    @staticmethod
    def get_mentions(sentence):
        return " ".join([word for word in sentence.split()
                         if word.startswith("@")])

    @staticmethod
    def remove_mentions(sentence):
        return " ".join([word for word in sentence.split()
                         if not word.startswith("@")])

    @staticmethod
    def get_hashtags(sentence):
        return " ".join([word for word in sentence.split()
                         if word.startswith("#")])

    @staticmethod
    def remove_hashtags(sentence):
        return " ".join([word for word in sentence.split()
                         if not word.startswith("#")])

    @staticmethod
    def remove_punct(sentence):
        text = " ".join([char for char in sentence.split()
                         if char not in string.punctuation])
        return re.sub('[0-9]+', '', text)

    @staticmethod
    def tokenization(sentence):
        return re.split('\W+', sentence)

    @staticmethod
    def remove_stopwords(sentence):
        return [word for word in sentence
                if word not in STOPWORDS_NLTK and len(word) > MIN_LEN_WORD]

    @staticmethod
    def stemming(sentence):
        ps = nltk.PorterStemmer()
        return [ps.stem(word) for word in sentence]

    @staticmethod
    def lemmatizer(sentence):
        wn = nltk.WordNetLemmatizer()
        return [wn.lemmatize(word) for word in sentence]

    def fit_transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        result = X.loc[:, self.tweets_column]\
            .apply(self.remove_urls)\
            .apply(self.remove_hashtags)\
            .apply(self.remove_mentions)\
            .apply(self.remove_punct).str.lower()\
            .apply(self.tokenization)\
            .apply(self.remove_stopwords)\
            .apply(self.stemming)\
            .apply(self.lemmatizer) + X.text.apply(self.get_hashtags).apply(
            wordninja.split)

        return result


class Modeling:
    def __init__(self, tweets_column):
        self.tweets_column = tweets_column
        self.model = None
        self.predictions = []
        self.test_rows = []
        self.le = None

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            stratify=y,
                                                            random_state=42)
        self.le = LabelEncoder()
        self.test_rows = X_test.index.tolist()

        clf = linear_model.SGDClassifier(n_jobs=-1, max_iter=1000, tol=1e-4,
                                         n_iter=None)
        model_calibrated = CalibratedClassifierCV(base_estimator=clf, cv=3,
                                                  method='sigmoid')

        self.model = make_pipeline(TfidfVectorizer(min_df=0.,
                                                   max_df=1.,
                                                   use_idf=True,
                                                   max_features=
                                                   20000),
                                   model_calibrated)

        self.model.fit(X_train.loc[:,
                       self.tweets_column].str.join(" "),
                       self.le.fit_transform(y_train))

    def predict(self, X):
        self.predictions = self.model.predict(
            X.loc[self.test_rows, self.tweets_column].str.join(" "))

    def score(self, y_true, y_pred):
        return classification_report(self.le.transform(y_true),
                                     y_pred,
                                     target_names=self.le.classes_)
