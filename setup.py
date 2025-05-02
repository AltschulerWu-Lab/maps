from setuptools import setup, find_packages

setup(
    name="maps",  # Replace with your package name
    version="0.0.0",  # Update as needed
    packages=['maps'],  # Automatically finds all subpackages
    install_requires=[
        'polars', 
        'pandas', 
        'seaborn', 
        'matplotlib', 
        'scikit-learn', 
        'numpy',
        'imblearn',
        'statsmodels'
    ],
    include_package_data=True,  # Includes non-code files (if using MANIFEST.in)
    author="Karl Kumbier",
    author_email="karl.kumbier@ucsf.edu",
    description="Pipeline for running ALS MAP scoring and related analyses",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",  # Change if applicable
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Change if needed
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",  # Set your minimum Python version
)
