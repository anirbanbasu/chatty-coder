# Python dependencies

You will need Python installed on your computer. The code in this repository has been tested on Python 3.12.0. Refrain from using the system Python. Use a Python version and virtual environment manager such as [pyenv](https://github.com/pyenv/pyenv). Create a new Python virtual environment for Python 3.12.0 or above. For example, this is how you do it using `pyenv`. The second line activates the virtual environment.

```zsh
# Uncomment the following line to install Python 3.12.0
# pyenv install 3.12.0
pyenv virtualenv 3.12.0 chatty-coder
pyenv activate chatty-coder
```

Once the virtual environment is setup, you can install all the dependencies for this application in it by running the following.

```zsh
pip install -U pip
pip install -U -r requirements.txt
```
## Optional: Uninstall all dependencies to start afresh

If necessary, you can uninstall everything previously installed by `pip` (in a virtual environment) by running the following.

```zsh
pip freeze | xargs pip uninstall -y
```
