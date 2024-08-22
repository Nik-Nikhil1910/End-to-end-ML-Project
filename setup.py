import setuptools

with open("C:/Users/asus/OneDrive/Desktop/Pandas app/End-to-end-ML-Project/README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "End-to-end-ML-Project-Implementation"
AUTHOR_USER_NAME = "Nik-Nikhil1910"
SRC_REPO = "mlProject"
AUTHOR_EMAIL = "nikhilkondinya@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author="Nikhil Sharma",
    author_email="nikhilkondinya@gmail.com",
    description="A small python package for ml app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/Nik-Nikhil1910/End-to-end-ML-Project.git",
    project_urls={
        "Bug Tracker": f"https://github.com/Nik-Nikhil1910/End-to-end-ML-Project/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)