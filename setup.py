from setuptools import setup

setup(
    name='d4g_dhs',
    version='0.1',
    packages=['d4g_dhs'],
    description='Detect Hate Speech library',
    include_package_data=True,
    install_requires=[
        'pandas',
        'scikit-learn',
        'nltk',
        'matplotlib',
        'wordcloud',
        'joblib',
        'wordninja',
        'xgboost',
        'streamlit'
    ],
)
