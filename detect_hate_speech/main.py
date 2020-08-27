import pandas as pd
import joblib

from .utils import (
    DataPreProcessing, GetWordCloud, Modeling
)

DATASET_PATH = "../data/detect_hate_speech_data.csv"


def main():
    data = pd.read_csv(DATASET_PATH,
                       sep='|',
                       index_col=0).set_index('tweet_id')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    data_prep = DataPreProcessing(tweets_column='text')
    X['clean_text'] = data_prep.fit_transform(X)

    # Modeling
    final_model = Modeling(tweets_column='clean_text')
    final_model.fit(X, y)

    joblib.dump(final_model, '../models/modeling.joblib')

    final_model.predict(X)
    print(final_model.predictions)
    print(final_model)
    print(final_model.score(y[final_model.test_rows], final_model.predictions))
    print("-" * 55)

if __name__ == "__main__":
    main()
