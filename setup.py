from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('development_requirements.txt') as f:
    dev_requirements = f.read().splitlines()

setup(
    name='mitiq',
    version='0.0.0',
    install_requires=requirements,
    extras_require={
        'development': dev_requirements
    },
    long_description=open('README.md').read(),
)
