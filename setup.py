from setuptools import setup, find_packages
from pip._internal.req import parse_requirements

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements("./requirements.txt", session='hack')

# reqs is a list of requirement
# e.g. ['django==1.5.1', 'mezzanine==1.4.6']
reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='dp-fasttext',
    version='0.9.0',
    packages=find_packages(),
    url='',
    license='',
    author='ONSDigital',
    author_email='',
    description='',
    install_requires=reqs
)
