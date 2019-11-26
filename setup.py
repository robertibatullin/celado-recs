from setuptools import setup

setup(name='celadorecs',
      version='0.1',
      description='Celado Recommender System',
      url='https://github.com/robertibatullin/celado-recs',
      author='CeladoAI',
      author_email='r.ibatullin@celado-media.ru',
      packages=['celadorecs'],
      scripts=['bin/celadorecs'],
      install_requires=[
          'flask', 
          'pandas',
          'numpy',
          'scipy',
          'scikit-learn'
      ],
      zip_safe=False)
