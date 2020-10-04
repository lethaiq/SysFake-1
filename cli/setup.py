from setuptools import setup

setup(
    name="SysFake CLI",
    description="A command-line interface for the SysFake fake news classifiers.",
    author="Terence Li and Hunter S. DiCicco",
    maintainer="Hunter S. DiCicco",
    version="1.2.8",
    url='https://github.com/dicicch/SysFake/',
    py_modules=['feature_extraction', 'sfake'],
    include_package_data=True,
    install_requires=['pandas', 'numpy', 'nltk',
                      'torch', 'click', 'scikit-learn',
                      'transformers', 'pyfiglet', 'goose3',
                      'language_check', 'tldextract', 'pyapa',
                      'waybackmachine'],
    data_files=[('models', ['ksgd-BERT.pickle',
                            'ksgd-taxonomy.pickle',
                            'sgd-BERT.pickle',
                            'sgd-taxonomy.pickle',
                            'sgd-tfidf.pickle']),
                ('models/tfidf', ['tfidf.pickle'])
                ]
)