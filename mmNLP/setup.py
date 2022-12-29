from setuptools import setup

setup(
    name='mmNLP',
    version='1.0.0',
    packages=['mmNLP'],
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
    ],
    author='M.M.',
    description=': A package to simplify the task of natural language processing',
    url='https://github.com/MMimoM/mmNLP',
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)