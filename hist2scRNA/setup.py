"""
Setup script for hist2scRNA package
"""

from setuptools import setup, find_packages
import os


def read_requirements():
    """Read requirements from requirements.txt"""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []


def read_readme():
    """Read README file"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''


setup(
    name='hist2scRNA',
    version='0.1.0',
    description='State-of-the-art model for single-cell RNA-seq prediction from histopathology images',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/hist2RNA',
    packages=find_packages(exclude=['tests', 'docs', 'notebooks']),
    install_requires=read_requirements(),
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='single-cell RNA-seq histopathology deep-learning vision-transformer spatial-transcriptomics',
    entry_points={
        'console_scripts': [
            'hist2scrna-train=hist2scRNA.train:main',
            'hist2scrna-evaluate=hist2scRNA.evaluate:main',
        ],
    },
)
