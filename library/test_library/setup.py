# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))


# This call to setup() does all the work
setup(
    name="cr_nlp",
    version="0.1.6",
    description="Library for NLP tasks by Aryan Oberoi",
    url="https://github.com/aryanoberoi/IITD-work/tree/main/library/test_library",
    author="Aryan Oberoi",
    author_email="aryanoberoi267@gmail.com",
    license="MIT",
    long_description="Library for NLP tasks by Aryan Oberoi",
    long_description_content_type='text/markdown',
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages=[""],
    include_package_data=True,
    install_requires=["transformers","nltk"]
)