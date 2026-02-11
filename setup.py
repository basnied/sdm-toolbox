from setuptools import setup, find_packages

setup(
    name='toolbox',
    version='0.1.0',
    author='Sebastian Nied',
    packages=find_packages(include=['toolbox', 'toolbox.*']),
)