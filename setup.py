from setuptools import setup, find_packages

setup(
    name='audiosentinel',
    version='0.1.0',
    author='Light',
    description='Human vs AI audio detection via Shannon entropy features',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_data={'audiosentinel': ['models/*.pkl']},
    install_requires=[
        'numpy>=1.24',
        'scipy>=1.10',
        'librosa>=0.10',
        'scikit-learn==1.6.1',
        'joblib>=1.3',
        'soundfile>=0.12',
    ],
    entry_points={
        'console_scripts': [
            'audiosentinel=main:main',
        ],
    },
    python_requires='>=3.9',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
    ],
)

