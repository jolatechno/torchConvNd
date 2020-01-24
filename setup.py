from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

def parse_requirements(filename):
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

setup(
    name="torchConvNd",
    version="0.2.0",
    author="Joseph TOUZET",
    author_email="joseph.touzet@gmail.com",
    description="a library to create convolution from any torch network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jolatechno/torchConvNd",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
