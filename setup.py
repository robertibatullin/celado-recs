from setuptools import setup

setup(name='celado-recs',
      version='0.1',
      description='Celado Recommender System',
      url='',
      author='CeladoAI',
      author_email='r.ibatullin@celado-media.ru',
      packages=['celado-recs'],
      install_requires=[
          'flask', 'os', 'webbrowser', 
          'threading', 
          'shutil',
          'warnings',
          'datetime',
          'pickle',
          'pandas',
          'numpy',
          'scipy',
          'scikit-learn'
      ],
      zip_safe=False)
