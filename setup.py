from setuptools import setup, find_packages


setup(
    name="prism",
    version="0.1",
    packages=find_packages(),
    entry_points={
        'console_scripts': {
            'run = prism.scripts.main:main',
        }
    },
)
