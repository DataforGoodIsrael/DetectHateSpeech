import wordninja
import nltk
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import string


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


class FeatureEngineering:
    pass


class GetWordCloud:
    def __init__(self, tweets_column,
                 background_color="white",
                 max_font_size=50,
                 max_words=100):
        self. tweets_column = tweets_column
        self.background_color = background_color
        self.max_font_size = max_font_size
        self.max_words = max_words

    def generate(self, X, y):
        fig, ax = plt.subplots(len(y.unique()), 1, figsize=(30, 30))
        for i, label in enumerate(y.unique()):
            df = X[y == label]
            tweets = " ".join(df.loc[:, self.tweets_column].str.join(" "))
            wordcloud = WordCloud(
                max_font_size=self.max_font_size,
                max_words=self.max_words,
                background_color=self.background_color).generate(tweets)
            ax[i].imshow(wordcloud, interpolation='bilinear')
            ax[i].set_title(str(label) + ' tweets', fontsize=30)
            ax[i].axis('off')
        return fig


class Modeling:
    def __init__(self, tweets_column):
        self.tweets_column = tweets_column
        self.pipelines = {}
        self.predictions = {}
        self.test_rows = []

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            stratify=y,
                                                            random_state=42)
        self.le = LabelEncoder()
        self.test_rows = X_test.index.tolist()
        self.pipelines['mnb'] = make_pipeline(CountVectorizer(),
                                              TfidfTransformer(),
                                              MultinomialNB())

        self.pipelines['lr'] = make_pipeline(CountVectorizer(),
                                             TfidfTransformer(),
                                             LogisticRegression(
                                                 class_weight='balanced',
                                                 solver='lbfgs',
                                                 multi_class='ovr')
                                             )

        self.pipelines['xgb'] = make_pipeline(CountVectorizer(),
                                              TfidfTransformer(),
                                              XGBClassifier(n_estimators=100,
                                                            n_jobs=4))

        # Add your idea for pipeline here

        for _, ppl in self.pipelines.items():
            ppl.fit(X_train.loc[:, self.tweets_column].str.join(" "),
                    self.le.fit_transform(y_train))

    def predict(self, X):
        for model, ppl in self.pipelines.items():
            self.predictions[model] = ppl.predict(
                X.loc[self.test_rows, self.tweets_column].str.join(" "))

    def score(self, y_true, y_pred):
        return classification_report(self.le.transform(y_true),
                                     y_pred,
                                     target_names=self.le.classes_)


def main():
    path = "../data/detect_hate_speech_data.csv"
    data = pd.read_csv(path, sep='|', index_col=0).set_index('tweet_id')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    data_prep = DataPreProcessing(tweets_column='text')
    X['clean_text'] = data_prep.fit_transform(X)

    # Feature Engineering: todo

    # Word Cloud
    gwc = GetWordCloud(tweets_column='clean_text')
    fig = gwc.generate(X, y)

    # Modeling
    modeling = Modeling(tweets_column='clean_text')
    modeling.fit(X, y)
    modeling.predict(X)
    for model, pred in modeling.predictions.items():
        print(model)
        print(modeling.score(y[modeling.test_rows], pred))
        print("-" * 55)

    # Output the best model
    import joblib
    joblib.dump(modeling, '../models/modeling.joblib')


if __name__ == "__main__":
    main()
