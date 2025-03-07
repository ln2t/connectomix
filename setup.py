from setuptools import setup, find_packages

# Function to read the contents of the requirements file
def read_requirements():
    with open('requirements.txt') as req:
        return req.read().splitlines()

setup(
    name='connectomix',
    version='2.1.0',
    url='https://github.com/ln2t/connectomix',
    author='Antonin Rovai',
    author_email='antonin.rovai@hubruxelles.be',
    description='nilearn-based BIDS app for functional connectivity fMRI data analysis (participant and group level) ',
    packages=find_packages(),
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'connectomix = connectomix.connectomix:main',
        ]},
    include_package_data=True,
    package_data={
            'connectomix': ['data/**'],
        }
)
