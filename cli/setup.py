import pathlib

from setuptools import setup
from os import path

# get the setup script's parent dir
HERE = pathlib.Path(__file__).parent

# later we will use the `readme` as the basis for the long description
#README = (HERE / "README.md").read_text()

with open(path.join(HERE, 'requirements.txt'), encoding='utf-8') as filein:
    ALL_REQS = filein.read().split('\n')

setup(
    name="sfake",
    description="A command-line interface for the SysFake fake news classifiers.",
    author="Terence Li and Hunter S. DiCicco",
    maintainer="Hunter S. DiCicco",
    version="1.2.9",
    url='https://github.com/dicicch/SysFake/',
    dependency_links=['https://github.com/pytorch/pytorch', 'https://github.com/pytorch/vision'],
    py_modules=['feature_extraction', 'sfake'],
    entry_points=dict(console_scripts='sfake=sfake.py:predict'),
    include_package_data=True,
    install_requires=ALL_REQS
)