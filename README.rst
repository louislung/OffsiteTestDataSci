OffsiteTestDataSci
******************

Install Anaconda / Miniconda
----------------

Install `anaconda <https://www.anaconda.com/download/>`
or `miniconda <https://conda.io/miniconda.html>`

Create virtual environment
--------------------------

Run the following using anaconda prompt

.. code:: bash
  
    conda create -n offsitetest_py275 python=3.7
    source activate offsitetest_py275

Install necessary packages
--------------------------

.. code:: bash
    
    source activate offsitetest_py275
    pip install -r requirements.txt
