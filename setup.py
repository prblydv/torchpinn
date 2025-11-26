from setuptools import setup, find_packages
 
setup(  
    name='torchpinn',
    version='0.1.0',
    description='A production-ready PyTorch library for Physics-Informed Neural Networks',
    author='Your Name', 
    packages=find_packages(),


    install_requires=['torch', 'numpy', 'matplotlib', 'scipy'], 
)