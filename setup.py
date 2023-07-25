from setuptools import setup, find_packages
from pkg_resources import parse_requirements

with open('requirements.txt') as f:
    requirements = [str(req) for req in parse_requirements(f)]


setup(
    name='flow_n_corr_utils',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    author='Shahar Zuler',
    author_email='shahar.zuler@gmail.com',
    description='A package that analyses 3D flow and 3D correspondence results, visualizes and provides common tools to process them.',
    url='https://github.com/shaharzuler/flow_n_corr_utils',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)