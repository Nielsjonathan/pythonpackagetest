from setuptools import setup, find_packages

setup(
    name='funda',
    version='0.1.0',
    description='Transformers package for Funda ML',
    packages=find_packages(include=['modules']),
    install_requires=[
        "ipykernel",
        "pandas",
        "numpy",
        "matplotlib",
        "plotnine",
        "scikit-learn"
        "pyarrow"
    ],
    python_requires="==3.8.5"
)