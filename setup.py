from setuptools import setup, find_packages

# Function to parse requirements.txt
def parse_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="my_pkg",
    version="0.1.0",
    description="A sample Python package requiring Python 3.10",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/my_package",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    extras_require={
        "dev": ["pytest", "flake8"],  # Optional development dependencies
    },
    python_requires=">=3.10",  # Specify minimum Python version
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
