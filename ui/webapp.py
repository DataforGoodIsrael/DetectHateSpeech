import streamlit as st
import joblib
import os
from detect_hate_speech import Modeling

NLP_MODEL = joblib.load(os.path.join('..', 'models', 'modeling.joblib'))

txt = st.text_area('Text to analyze', '''
It was the best of times, it was the worst of times, it was
the age of wisdom, it was the age of foolishness, it was
the epoch of belief, it was the epoch of incredulity, it
was the season of Light, it was the season of Darkness, it
was the spring of hope, it was the winter of despair, (...)
''')

nlp_model_result = NLP_MODEL.le.classes_[NLP_MODEL.pipelines['xgb'].predict([txt])[0]]
st.write('Sentiment:', nlp_model_result)

