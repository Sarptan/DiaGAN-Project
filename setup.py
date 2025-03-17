from setuptools import setup, find_packages

setup(
    name="diagan_project",
    version="0.1",
    packages=find_packages(where="src"),  # Finds all packages inside src/
    package_dir={"": "src"},  # Declares that src/ is the root for packages
)