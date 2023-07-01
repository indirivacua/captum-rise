from setuptools import setup

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

url = 'https://github.com/indirivacua/captum-rise'

# python setup.py sdist
setup(
    name='captum-rise',
    version='1.0',
    python_requires='>=3.8',
    install_requires=[
        'torch',
        'captum'
    ],
    py_modules=['rise'],
    
    # metadata to display on PyPI
    author='Oscar Agustín Stanchi',
    author_email='oscar.stanchi@gmail.com',
    description='The implementation of the RISE algorithm for the Captum framework',
    keywords='Black-Box Computer-Vision Deep-Learning Interpretability RISE',
    url=url,
    project_urls={
        "Bug Tracker": url+"/issues",
        "Documentation": url,
        "Source Code": url,
    },
    license='BSD-3-Clause',
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.8',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
)
