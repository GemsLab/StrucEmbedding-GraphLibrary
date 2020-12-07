import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as freq:
    requirements = freq.read().splitlines()

setuptools.setup(
    name="semb",
    version="0.0.1",
    author="The SEMB Team, University of Michigan",
    author_email="jackierw@umich.edu",
    description="SEMB client side API library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ReactiveXYZ-Dev/StructuralEmbeddingUnderstanding",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    # install_requires=requirements
)
