from setuptools import setup, find_packages

setup(
    name="maps",  # Replace with your package name
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'polars', 
        'pandas', 
        'seaborn', 
        'matplotlib', 
        'scikit-learn', 
        'numpy',
        'imblearn',
        'statsmodels',
        'torch>=2.1.0'
    ],
    include_package_data=True,
    author="Karl Kumbier",
    author_email="karl.kumbier@ucsf.edu",
    description="Pipeline for running ALS MAP scoring and related analyses",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",  # Change if applicable
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",  # Set your minimum Python version
)
