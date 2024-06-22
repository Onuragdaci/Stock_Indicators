from setuptools import setup, find_packages

setup(
    name='Stock_Indicator',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'tradingview-screener',
        'git+https://github.com/rongardF/tvdatafeed'
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='Package for stock indicators',
    url='https://github.com/yourusername/Stock_Indicator',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
