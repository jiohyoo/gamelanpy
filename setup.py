from distutils.core import setup

setup(
    name='gamelanpy',
    version='0.1.1',
    author='Ji Oh Yoo',
    author_email='jioh.yoo@gmail.com',
    packages=['gamelanpy'],
    scripts=['scripts/gamelan_learn_model.py', 'scripts/gamelan_imputation.py'],
    license='LICENSE',
    description='Python Implementation of GAMELAN core',
    install_requires=[
        "numpy >= 1.6.0",
        "scipy >= 0.13.0",
        "scikit-learn >= 0.16.0"
    ],
)
