# Detect Hate Speech

Model that target Homophobic and Sexist Tweets to be reported to Twitter.

### Data

### Solution

### Instalation


### Webapp


Active Learning
###############

A small pure-python package for active learning at Zencity

.. code-block:: python

  from active_learning.api import get_active_learning_samples
  from active_learning.interfaces import (
    ActiveLearningRequest,
    ActiveLearningAlgorithm,
    ActiveLearningSampleSelection
  )

  active_learning_request = ActiveLearningRequest(
        language='en',
        lastKdays=1,
        algorithm=ActiveLearningAlgorithm.satisfaction,
        item_sub_type='post',
        k=50,
        sample_selection=ActiveLearningSampleSelection.margin,
    )

  results = get_active_learning_samples(active_learning_request)


.. contents::


.. section-numbering::


Installation
============

First, make sure your ``pip`` is configured to also use ZenCity's private PyPI server hosted on Ido's Gemfury account.

The, install ``active learning`` with:

.. code-block:: bash

  pip install active-learning

Configuration
=============

The package uses birch to read MongoDB-related configuration. To access ZenCity's MongoDB server you have to provide utilizen with the full connection uri and the database name, either through a configuration file or environment variables.

To provide them using a configuration file, create a .zcmongo directory in your home folder, and populate it a cfg.json file with the following content:

.. code-block:: json

    {
        "ZCDB_AL_URI": "<production uri goes here>",
        "ZCDB_AL_NAME": "zcdb_name"
    }

Alternatively, you can definde the ZCMONGO_ZCDB_AL and ZCMONGO_ZCDB_AL_NAME environment variables with corresponding values.

Contributing
============

Package author and current maintainer are Ori Cohen and Samuel Jefroykin;
You are more than welcome to approach him for help.
Contributions are very welcomed.

Installing for development
--------------------------

Clone:

.. code-block:: bash

  git clone git@bitbucket.org:zencitytech/active-learning.git


Install in development mode with test dependencies:

.. code-block:: bash

  cd active-learning
  pip install -e ".[test]"


Running the tests
-----------------

To run the tests, use:

.. code-block:: bash

  python -m pytest --cov=active_learning --doctest-modules


Adding documentation
--------------------

This project is documented using the `numpy docstring conventions`_, which were chosen as they are perhaps the most widely-spread conventions that are both supported by common tools such as Sphinx and result in human-readable docstrings (in my personal opinion, of course). When documenting code you add to this project, please follow `these conventions`_.

.. _`numpy docstring conventions`: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _`these conventions`: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt


Credits
=======
Created by Ori Cohen and Samuel Jefroykin
ori@zencity.io, samuel@zencity.io


