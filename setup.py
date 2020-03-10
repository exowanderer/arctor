try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    from distutils.core import setup, find_packages, Extension

# Copyright (c) 2015-2019 Patricio Cubillos and contributors.
# MC3 is open-source software under the MIT license (see LICENSE).

from numpy import get_include

inc = [get_include()]
eca = ['-ffast-math']
ela = []

ext1 = Extension('arctor.dwt._dwt',
                 sources=['arctor/dwt/_dwt.c'],
                 include_dirs=inc,
                 extra_compile_args=eca,
                 extra_link_args=ela)
extensions = [ext1]

setup(name='arctor',
      version=0.1,
      description='Extracting Photometry from Scanning Mode HST Observations '
      ' and other arc-like observations that require a rectangular aperture',
      long_description=open('README.md').read(),
      url='https://github.com/exowanderer/arctor',
      license='GPL3',
      author="Jonathan Fraine (exowanderer) and STARGATE",
      packages=find_packages(),
      install_requires=['joblib', 'numpy', 'pandas', 'astropy',
                        'photutils', 'scipy', 'statsmodels', 'tqdm',
                        'requests'],
      extras_require={'plots':  ["matplotlib"]},
      ext_modules=extensions
      )
