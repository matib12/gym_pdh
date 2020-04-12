from setuptools import setup

setup(name='gym_pdh',
      version='0.0.1',
      description='gym of Pound-Drever-Hall systems',
      author='Mateusz Bawaj',
      author_email='mateusz.bawaj@pg.infn.it',
      install_requires=['gym', 'numpy', 'pdh', 'Pillow']#All dependencies required. pdh Mateusz's package
      #packages=['gym', 'numpy', 'Pillow']#All dependencies required. pdh Mateusz's package
)
