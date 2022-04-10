from setuptools import setup, find_packages

setup(
    name='pic',
    version='0.0.2',
    description='pytorch image classification',
    url='https://github.com/hankyul2/pytorch-image-classification',
    packages=find_packages(exclude=['tests']),
    author='hankyul',
    author_email='consistant1y@ajou.ac.kr',
)