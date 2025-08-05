from setuptools import setup, find_packages

setup(
    name='loop',
    version='1.5.0',
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "scikit-learn>=1.0.0",
        "ta>=0.10.0",
        "tensorflow>=2.12.0",
        "keras_tuner>=1.1.0",
        "seaborn>=0.12.0",
        "numpy>=1.22.0",
        "matplotlib>=3.5.0",
        "plotly>=5.14.0",
        "dash>=2.11.0",
        "dash-bootstrap-components>=1.4.0",
    ],
    python_requires='>=3.10',
    author='Mikko Kotila',
    author_email='mikko@empiricalusa.com',
    description='Bitcoin-first research and trading platform.',
)
