Detect Hate Speech
###############

A small solution for targeting Homophobic and Sexist Tweets to be reported to Twitter by Data For Good, Israel.


Data Collection
===============
We merge multiple dataset:

- We are using hate speech dataset from https://github.com/ZeerakW/hatespeech and https://data.world/thomasrdavidson/hate-speech-and-offensive-language
- We are scrapping sexist and homophobic tweets thanks to hashtags and doing special annotation.

Solution
========

A Text Classification Xgboost Model.

Webapp
======


Installation
============

.. code-block:: bash

  pip install -r requirements.txt


Contributing
============

Author and current maintainer are the Data For Good Tema.
You are more than welcome to approach us for help.
Contributions are very welcomed.

Installing for development
--------------------------

Clone:

.. code-block:: bash

  git clone https://github.com/DataforGoodIsrael/DetectHateSpeech.git


Credits
=======
Created by Jeremy Atias and Samuel Jefroykin from Data For Good Israel
hello@dataforgoodisrael.com
