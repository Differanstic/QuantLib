from setuptools import setup, find_packages
with open("requirements.txt") as f:
    requirements = f.read().splitlines()
setup(
    name="Quantlib",
    version="1.1.0",
    author="SadalSuud",
    author_email="kathancpandya@gmail.com",
    description="Quant Library for Market Analysis and Automation",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.9",
)
