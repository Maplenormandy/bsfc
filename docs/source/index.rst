Bayesian Spectral Fitting Code (BSFC)
=====================================

.. image:: complex_bsfc_fit.png
   :width: 800px
   :align: center
   :target: complex_bsfc_fit.html

	    
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   api
 

The Bayesian Spectral Fitting Code (BSFC) is a tool to compute moments and quantify uncertainties in high-resolution X-ray plasma spectroscopy. Its initial intended purpose was to improve standard fitting routines based on non-linear optimizers for X-Ray Imaging Crystal Spectroscopy (XICS) at the Alcator C-Mod tokamak at MIT. However, its methods are not specific to this instruments, or to tokamaks altogether, and may be applied to other contexts.

:py:mod:`bsfc` combines two elements of innovation:

#. The use of an elegant truncation for an Hermite polynomial decomposition of quasi-Gaussian atomic lines, which is particularly appropriate for spectrally-resolved line shapes that result from line integrated signals, which are not well described by Gaussians. ``bsfc``'s polynomial decomposition is very effective in separating overlapping atomic lines, which are just as hard as they are common in plasma spectroscopy.
   
#. The application of multiple Bayesian techniques to explore parameters that describe well the provided data. In particular, we have found nested sampling (NS) methods to outperform other MCMC-like methods, particularly for what concerns the process of ``model selection``, which is here used to choose the most appropriate truncation to the polynomial decomposition for each line.

The ``emcee``, ``MultiNest``, ``PolyChord`` and ``dyPolyChord`` sampling algorithms are all implemented, but users require their own installation to make them work.

    *By making the BSFC code open-source and documented we hope to make it easier for others to have a closer look at the adopted methods to advance other applications in plasma spectroscopy (and other fields?).*

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



Attribution
------------
If you find :py:mod:`bsfc` and its open-source code useful for your academic research, please cite our paper:

.. code-block:: tex

    @article{bsfc2020,
    author={Cao, Norman M. and Sciortino, Francesco},
    title={Bayesian Spectral Moment Estimation and Uncertainty Quantification},
    year={2020},
    volume={48},
    number={1},
    pages={22-30},
    journal={IEEE Transactions on Plasma Science},
    doi={ 10.1109/TPS.2019.2946952},
    url={https://ieeexplore.ieee.org/abstract/document/8879689}}


Contributions
-------------
Contributions are welcome -- please use the Github page to create pull requests or point out issues:
https://github.com/Maplenormandy/bsfc
For any questions, please email sciortino@psfc.mit.edu



Authors & License
-----------------

Copyright 2020-Present Francesco Sciortino and Norman M. Cao (MIT license).


