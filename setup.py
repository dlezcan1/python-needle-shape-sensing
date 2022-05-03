from setuptools import setup

setup(
        name='needle-shape-sensing',
        version='0.3.6',
        author='Dimitri Lezcano',
        author_email='dlezcan1@jhu.edu',
        packages=[ 'needle_shape_sensing' ],
        url='http://pypi.python.org/pypi/needle-shape-sensing/',
        license='LICENSE.txt',
        description='Needle Shape Sensing library',
        long_description=open( 'README.md' ).read(),
        install_requires=[
                'numpy',
                'scipy',
                'spatialmath-python',
                'sympy',
                'numba'
                ],
        classifiers=[
                'Operating System :: OS Independent',
                'License :: OSI Approved :: MIT License'
                ]
        )
