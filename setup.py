from setuptools import setup, find_packages

setup(
    name='segment-anything-fast',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'diskcache',
        'pycocotools',
        'scipy',
        'scikit-image',
    ],
    description='A pruned, quantized, compiled, nested and batched implementation of nested tensor',
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/segment-anything-fast',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
