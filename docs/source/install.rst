.. _install:

Installation
============

``bsfc`` is compatible with python 2.7 and 3+, and can be installed with `pip <http://www.pip-installer.org/>`_:

.. code-block:: bash

   pip install bsfc


Alternatively, you can download the latest version and by cloning `the github
repository <https://github.com/Maplenormandy/bsfc>`_:

.. code-block:: bash

    git clone https://github.com/ejhigson/dyPolyChord.git
    cd bsfc
    python setup.py install

Note that the github repository may include new changes which have not yet been released on PyPI (and therefore will not be included if installing with pip).

``bsfc`` is parallelized via Python ``multiprocessing`` with shared memory (within 1 machine/node) and can use MPI when combined with ``MultiNest`` or ``PolyChord``. In order to use MPI, make sure to install ``MultiNest`` and ``PolyChord`` with MPI, following the instructions given at the links below for each package.


Dependencies
------------

``bsfc`` requires:

 - ``MultiNest`` >=v3.10;
 - ``pyMultiNest`` 
 - ``emcee`` >=v2.2.1;
 - ``PolyChord`` >=v1.14;
 - ``numpy`` >=v1.13;
 - ``scipy`` >=v1.0.0;
 - ``nestcheck`` >=v0.1.8.


Note that in order to use ``BSFC`` the codes above must be individually installed by the user. Refer to the following pages to find out how:

* https://github.com/PolyChord/PolyChordLite

* https://github.com/ejhigson/dyPolyChord

* https://github.com/JohannesBuchner/PyMultiNest/tree/master/pymultinest

* https://emcee.readthedocs.io/en/stable/

Note that ``emcee`` is not well maintained within our code since we don't use it much at this time, but it should work in principle.


