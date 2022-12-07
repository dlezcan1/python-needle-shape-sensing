from setuptools import setup, find_packages

setup(
        name='needle-shape-sensing',
        version='0.7.2',
        author='Dimitri Lezcano',
        author_email='dlezcan1@jhu.edu',
        packages=find_packages(),
        url='http://pypi.python.org/pypi/needle-shape-sensing/',
        license='LICENSE.txt',
        description='Needle Shape Sensing library',
        long_description=open( 'README.md' ).read(),
        install_requires=[
                'numpy',
                'scipy',
                'spatialmath-python',
                'sympy',
                'numba',
                'tensorflow',
                'torch',
                'torchvision',
                'torchaudio',
                ],
        classifiers=[
                'Operating System :: OS Independent',
                'License :: OSI Approved :: MIT License'
                ]
        )
