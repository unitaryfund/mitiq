from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='mitiq',
    version='0.0.0',
    install_requires=requirements,
    long_description=open('README.md').read(),
)
