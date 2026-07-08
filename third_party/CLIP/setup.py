import os
from setuptools import find_packages, setup

def _parse_requirements(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

setup(
    name="clip",
    py_modules=["clip"],
    version="1.0",
    description="",
    author="OpenAI",
    packages=find_packages(exclude=["tests*"]),
    install_requires=_parse_requirements(os.path.join(os.path.dirname(__file__), "requirements.txt")),
    include_package_data=True,
    extras_require={"dev": ["pytest"]},
)
