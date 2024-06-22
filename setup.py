from setuptools import setup, find_packages

setup(
    name='StockIndicators',
    version='1.0',
    packages=find_packages(),
    author='Onur AĞDACI',
    author_email='onuragdaci@gmail.com',
    description='Library for stock indicators',
    url='https://github.com/Onuragdaci/stock_indicators',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=[
      'scipy',
      "numpy",
      "pandas",
      "tradingview-screener",
      "https://github.com/rongardF/tvdatafeed tradingview-screener"
    ],
)
