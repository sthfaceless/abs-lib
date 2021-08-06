from setuptools import setup

setup(
    name='abslib',
    version='1.0.5',
    description='Library for working with algebraic bayesian networks',
    license='MIT',
    packages=['abslib'],
    install_requires=[
        'numpy',
        'cvxopt',
    ],
    author='Tatiana Stelmakh',
    author_email='st087048@student.spbu.ru',
    keywords=['abs', 'statistic'],
    url='https://github.com/sthfaceless/abs-lib'
)
