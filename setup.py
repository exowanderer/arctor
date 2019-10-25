try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

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
      extras_require={'plots':  ["matplotlib"]}
      )
