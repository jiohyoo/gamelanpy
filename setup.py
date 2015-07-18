from distutils.core import setup, Extension
import numpy as np

ext_weighted_kmeans = Extension('_weighted_kmeans',
                                sources=['gamelanpy/_weighted_kmeans.c'],
                                include_dirs=[np.get_include()])

setup(
    name='gamelanpy',
    version='0.1.2',
    author='Ji Oh Yoo',
    author_email='jioh.yoo@gmail.com',
    packages=['gamelanpy'],
    scripts=['scripts/gamelan_learn_model.py', 'scripts/gamelan_imputation.py'],
    license='LICENSE.txt',
    description='Python Implementation of GAMELAN core',
    install_requires=[
        "numpy >= 1.6.0",
        "scipy >= 0.13.0",
        "scikit-learn >= 0.16.0"
    ],
    ext_modules=[ext_weighted_kmeans]
)