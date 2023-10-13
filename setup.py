from setuptools import setup, find_packages

packages = find_packages()
print("packages: ", packages)
setup(
    name='segment-anything-fast',
    version='0.1',
    packages=packages,
    install_requires=[
        'torch',
        'diskcache',
        'pycocotools',
        'scipy',
        'scikit-image',
    ],
    description='A pruned, quantized, compiled, nested and batched implementation of nested tensor',
    long_description_content_type='text/markdown',
    url='https://github.com/pytorch-labs/segment-anything-fast',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
