import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


with open("requirements.txt", "r") as fh:
    requirements = [x.strip("\n")
                    for x in fh.readlines()
                    if x]


setuptools.setup(
    name="conversational-sentence-encoder",
    version="0.0.6",
    author="David Alami",
    author_email="davidalami@gmail.com",
    description="Dual sentence encoder package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/davidalami/ConveRT",
    packages=setuptools.find_packages(),
    install_requires= requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7.0',
)
