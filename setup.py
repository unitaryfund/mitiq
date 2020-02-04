from distutils.core import setup

with open("VERSION.txt", "r") as f:
    __version__ = f.read().strip()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('development_requirements.txt') as f:
    dev_requirements = f.read().splitlines()

setup(
    name='mitiq',
    version=__version__,
    install_requires=requirements,
    extras_require={
        'development': dev_requirements
    },
    long_description=open('README.md').read(),
)
