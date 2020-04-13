from setuptools import setup, find_packages

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
        'development': set(dev_requirements),
        'test': dev_requirements,
    },
    packages = find_packages(),
    include_package_data=True,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author = "Unitary Fund",
    classifiers=[
         "Development Status :: 2 - Pre-Alpha",
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
         "Operating System :: MacOS",
         "Operating System :: Unix",
         "Operating System :: Microsoft :: Windows",
         "Topic :: Scientific/Engineering",
         ],
     license = "GPL v3.0",
     url = "https://unitary.fund",

)
