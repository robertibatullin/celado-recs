from setuptools import setup

setup(name='celadorecs',
      version='2.3',
      description='Celado Recommender System',
      url='https://github.com/robertibatullin/celado-recs',
      author='CeladoAI',
      author_email='r.ibatullin@celado-media.ru',
      packages=['celadorecs'],
      scripts=['bin/celadorecs'],
      install_requires=[
          'pandas',
          'numpy',
          'scipy',
          'scikit-learn'
      ],
      zip_safe=False,
      include_package_data=True)
