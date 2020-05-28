from setuptools import find_packages, setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='textformer',
      version='1.0.0',
      description='Transformer-based Text Operators',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Gustavo Rosa',
      author_email='gustavo.rosa@unesp.br',
      url='https://github.com/gugarosa/textformer',
      license='GPL-3.0',
      install_requires=['coverage>=5.1',
                        'pylint>=2.5.2',
                        'pytest>=5.4.2',
                        'torch>=1.5.0',
                        'torchtext>=0.6.0',
                        'tqdm>=4.46.0'
                        ],
      extras_require={
          'tests': ['coverage',
                    'pytest',
                    'pytest-pep8',
                    ],
      },
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
