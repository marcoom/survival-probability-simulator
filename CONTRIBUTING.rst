Contributing
============

Any contributions are welcome and appreciated!

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/marcoom/survival-probability-simulator/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with **bug** and **help wanted** is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the issues for features. Anything tagged with **enhancement**
and **help wanted** is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Survival Probability Simulator could always use more documentation, whether as part of the
official Survival Probability Simulator docs, in docstrings, or even on the web in blog posts,
articles, and such.

Documentation Style
:::::::::::::::::::

This project uses `Google Python Documentation Style <https://google.github.io/styleguide/pyguide.html>`_.

**Note**:

- For documenting endpoint functions, please use ``\f`` in your documentation to truncate the output used for OpenAPI at this point.

Please check `Advanced description from docstring <https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#advanced-description-from-docstring>`_ for more details.


Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/marcoom/survival-probability-simulator/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.

Get Started!
------------

Ready to contribute? Here's how to set up `survival_probability_simulator` for local development.

1. Fork the `survival_probability_simulator` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@your_repo_url.git

3. Install your local copy into a virtualenv. Assuming you have virtualenv installed, this is how you set up your fork for local development::

    $ python -m venv survival-probability-simulator-venv
    $ source survival-probability-simulator-venv/bin/activate
    $ cd survival-probability-simulator/

   Now you can install the dependencies in your virtual environment::

    $ pip install -r requirements.txt

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, test your changes by running the application locally::

    $ streamlit run app.py

   Make sure all visualizations load correctly and the prediction functionality works.

6. Run any data preprocessing or model training scripts if you modified them::

    $ python scripts/preprocess_data.py
    $ python scripts/train_model.py

7. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should maintain code quality and follow PEP 8 standards.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.

Docker Development
------------------

You can also develop using Docker::

    $ docker build -t titanic-survival-simulator .
    $ docker run -p 8501:8501 titanic-survival-simulator

The application will be available at http://localhost:8501