from setuptools import setup, find_packages

# Function to read the contents of the requirements file
def read_requirements():
    with open('requirements.txt') as req:
        return req.read().splitlines()

setup(
    name='connectomix',
    version='1.0.1',
    url='https://github.com/ln2t/connectomix',
    author='Antonin Rovai',
    author_email='antonin.rovai@hubruxelles.be',
    description='nilearn-based BIDS app for connectome analysis from rsfMRI',
    packages=find_packages(),
    install_requires=read_requirements(),
    package_data={'connectomix': ['data/*']},
    entry_points={
        'console_scripts': [
            'connectomix = connectomix.connectomix:main',
        ]},
    include_package_data=True
)
