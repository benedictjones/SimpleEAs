from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as f:
    long_description = f.read()



if __name__ == "__main__":
    setup(
        name="pyeas",
        version="0.1.0",
        author="benedictjones",
        description="Implements Evolutionary Algorithms and tools",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/benedictjones/pyeas",
        license="BSD License (BSD-3-Clause)",
        packages=find_packages(),
        keywords=['python', 'Differential Evolution', 'DE', 'OpenAI Evolutionary Strategy', "OpenAI ES"],
        install_requires=["numpy"],
        extras_requires=["twine"],
        )
