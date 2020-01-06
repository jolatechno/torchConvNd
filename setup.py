import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchGeneralizer", # Replace with your own username
    version="0.0.1",
    author="Joseph TOUZET",
    author_email="joseph.touzet@gmail.com",
    description="a library to create convolution from any torch network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jolatechno/torchGeneralizer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
