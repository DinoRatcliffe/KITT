import sys
from distutils.core import setup

import os
from setuptools import find_packages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kitt'))
from version import VERSION

extras = {
        'gym': ['gym[atari, box2d]'],
}

gpu_extras = {
    'tensorflow-gpu': ['tensorflow-gpu>=2.0.0'],
}

cpu_extras = {
    'tensorflow': ['tensorflow>=2.0.0'],
}

cpu_deps = sum(extras.values(), sum(cpu_extras.values(), []))
gpu_deps = sum(extras.values(), sum(gpu_extras.values(), []))

# cpu based extras are the default (should work everywhere)
extras['all'] = cpu_deps
extras['all-gpu'] = gpu_deps

setup(
        name='kitt',
        version=VERSION,

        packages=[package for package in find_packages()
                  if package.startswith('kitt')],

        license='GNU General Public License v2 (GPLv2)',
        author='Dino Stephen Ratcliffe',
        author_email='ratcliffe@dino.ai',
        description='Platform for AI experimentation',
        install_requires=['numpy',
                          'cma', 
                          'pandas', 
                          'nptyping', 
                          'protobuf==3.12.0',
                          'seaborn', 
                          'orjson',
                          'matplotlib', 
                          'scikit-image',
                          'tensorflow_probability', 
                          'dm-sonnet>=2.0.0',
                          'gast==0.3.3'],
        extras_require=extras,
        test_require=['mock>=2.0.0', 'pytest==4.1.1']
)
