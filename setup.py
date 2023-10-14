from setuptools import setup, find_packages

packages = find_packages()
print("packages: ", packages)
setup(
    name='segment-anything-fast',
    version='0.2',
    packages=packages,
    install_requires=[
        'torch',
        'diskcache',
        'pycocotools',
        'scipy',
        'scikit-image',
    ],
    data_files=[
        "segment_anything_fast/configs/flash_4_configs_a100.p",
        "segment_anything_fast/configs/int_mm_configs_a100.p",
    ],
    description='A pruned, quantized, compiled, nested and batched implementation of segment-anything',
    long_description_content_type='text/markdown',
    url='https://github.com/pytorch-labs/segment-anything-fast',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
