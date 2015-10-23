""" BHMM: A toolkit for Bayesian hidden Markov model analysis of single-molecule trajectories.

This project provides tools for estimating the number of metastable states, rate
constants between the states, equilibrium populations, distributions
characterizing the states, and distributions of these quantities from
single-molecule data. This data could be FRET data, single-molecule pulling
data, or any data where one or more observables are recorded as a function of
time. A Hidden Markov Model (HMM) is used to interpret the observed dynamics,
and a distribution of models that fit the data is sampled using Bayesian
inference techniques and Markov chain Monte Carlo (MCMC), allowing for both the
characterization of uncertainties in the model and modeling of the expected
information gain by new experiments.
"""

from __future__ import print_function
__requires__ = ['setuptools>=18',]
import os
from os.path import relpath, join

import versioneer
import pkg_resources
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as _build_ext

DOCLINES = __doc__.split("\n")

########################
CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)
Programming Language :: Python
Topic :: Scientific/Engineering :: Bio-Informatics
Topic :: Scientific/Engineering :: Chemistry
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

################################################################################
# USEFUL SUBROUTINES
################################################################################
def find_package_data(data_root, package_root):
    files = []
    for root, dirnames, filenames in os.walk(data_root):
        for fn in filenames:
            files.append(relpath(join(root, fn), package_root))
    return files


################################################################################
# SETUP
################################################################################

extensions = [Extension('bhmm.hidden.impl_c.hidden',
                        sources = ['./bhmm/hidden/impl_c/hidden.pyx',
                                   './bhmm/hidden/impl_c/_hidden.c'],
                        include_dirs = ['/bhmm/hidden/impl_c/']),
              Extension('bhmm.output_models.impl_c.discrete',
                        sources = ['./bhmm/output_models/impl_c/discrete.pyx',
                                   './bhmm/output_models/impl_c/_discrete.c'],
                        include_dirs = ['/bhmm/output_models/impl_c/']),
              Extension('bhmm.output_models.impl_c.gaussian',
                        sources = ['./bhmm/output_models/impl_c/gaussian.pyx',
                                   './bhmm/output_models/impl_c/_gaussian.c'],
                        include_dirs = ['/bhmm/output_models/impl_c/']),
              Extension('bhmm._external.clustering.kmeans_clustering',
                        sources=['./bhmm/_external/clustering/src/clustering.c',
                                 './bhmm/_external/clustering/src/kmeans.c'],
                        include_dirs=['./bhmm/_external/clustering/include',
                                     ],
                        extra_compile_args=['-std=c99']),
              ]


class build_ext(_build_ext):
    def initialize_options(self):
        _build_ext.initialize_options(self)
        import pkg_resources
        dir = pkg_resources.resource_filename('numpy', 'core/include')
        self.include_dirs = [dir]

cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = build_ext

setup(
    name='bhmm',
    author='John Chodera and Frank Noe',
    author_email='john.chodera@choderalab.org',
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    license='LGPL',
    url='https://github.com/bhmm/bhmm',
    platforms=['Linux', 'Mac OS-X', 'Unix', 'Windows'],
    classifiers=CLASSIFIERS.splitlines(),
    package_dir={'bhmm': 'bhmm'},
    packages=find_packages(),
    package_data={'bhmm': find_package_data('examples', 'bhmm') + find_package_data('bhmm/tests/data', 'bhmm')},  # NOTE: examples installs to bhmm.egg/examples/, NOT bhmm.egg/bhmm/examples/.  You need to do utils.get_data_filename("../examples/*/setup/").
    zip_safe=False,
    install_requires=[
        'numpy',
        'scipy',
        'msmtools',
        'six',
        ],
    setup_requires=[
        'cython',
        'numpy',
        ],
    ext_modules=extensions,
    )

