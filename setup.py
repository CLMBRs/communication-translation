from setuptools import setup, find_packages

with open("./requirements.txt") as f:
    requirements = f.read().splitlines()

# Package (minimal) configuration
setup(
    name="clmbrs-communication-translation",
    version="0.1.0",
    description=
    "Emergent Communication Fine-tuning for Pre-trained Language Model",
    packages=find_packages(),  # __init__.py folder search
    install_requires=requirements
)
