from setuptools import setup


setup(
    name='tsa-lth',
    version='1.0',
    description='Package for time series analysis course at LTH',
    author='Filipp Lernbo',
    author_email='fi6418le-s@student.lu.se',
    packages=['TSA'],
    install_requires=[
        "filterpy==1.4.5",
        "matplotlib==3.7.1",
        "nfoursid==1.0.1",
        "pandas==2.1.0",
        "scipy==1.11.0",
        "statsmodels==0.14.0",
    ],
)
