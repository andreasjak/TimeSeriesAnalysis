from setuptools import setup


setup(
    name='tsa-lth',
    version='1.0',
    description='Package for time series analysis course at LTH',
    author='Filipp Lernbo, Tadas',
    packages=['tsa_lth'],
    install_requires=[
        "filterpy",
        "matplotlib",
        "pandas",
        "scipy",
        "statsmodels",
        "numpy",
    ],
)
