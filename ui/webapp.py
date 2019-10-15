import streamlit as st
import joblib
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'detect_hate_speech'))
from detect_hate_speech import Modeling

NLP_MODEL = joblib.load(os.path.join('..', 'models', 'modeling.joblib'))

st.title('Detect Hate Speech')

st.write('The current model predicts the following classes', str(NLP_MODEL.le.classes_.tolist()))
txt = st.text_area('Text to analyze', ''' #Feminazi strike again!!''')

nlp_model_result = NLP_MODEL.le.classes_[NLP_MODEL.pipelines['xgb'].predict([txt])[0]]
st.write('**Sentiment**:', str(nlp_model_result))
