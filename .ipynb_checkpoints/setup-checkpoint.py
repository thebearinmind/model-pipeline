from setuptools import setup

setup(name='bearinmind_pipeline',
      version='0.0.1',
      description='The package for bloomberg message screening',
      url= 'https://github.com/thebearinmind/model-pipeline',
      author='Ihor Shylo',
      author_email='ihor.shylo@gmail.com',
      license='MIT',
      packages=['bearinmind_pipeline'],
      install_requires=[
          'lightgbm',
          'pandas',
          'numpy',
          'sklearn',
          'catboost',
          'xgboost',
          'matplotlib',
          'seaborn'
      ],
      zip_safe=False)