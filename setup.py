import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="astro_pola", # Replace with your own username
    version="0.0.1",
    author="Paula Cáceres Burgos",
    author_email="paula.caceres.burgos@gmail.com",
    description="astro_pola: astronomy related functions used along the way...",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PauCaBu/astro_pola",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.9.1',
)