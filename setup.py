from setuptools import setup, find_packages

setup(
    name='NovaSR',
    version='0.0.2',
    packages=find_packages(),
    author='ysharma3501',
    description='A fast, small, and high-quality neural audio upsampling model.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ysharma3501/NovaSR',
    license='MIT',
    install_requires=[
        'soxr',
        'timm',
        'torchaudio',
        'torch',
        'torchcodec',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
