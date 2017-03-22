#!/bin/bash

# proxy 
# export all_proxy=http://proxy.esrf.fr:3128/

# clean old stuff
echo "Cleaning old installation files..."
rm -rf tmp1env


#
# step 1: create and start python3 virtual environment
#

virtualenv -p python3 tmp1env
source tmp1env/bin/activate

#
# step 2: Upgrade tools
#

python -m pip install --upgrade pip

pip install --upgrade setuptools

pip install --upgrade wheel



#
# step 3: install numpy and scipy
#

echo "Installing numpy"
pip install numpy
echo "Installing scipy"
pip install scipy

#git clone https://github.com/srio/und_Sophie_2016
#cd und_Sophie_2016
python setup.py install


python pySRU/tests/AllTests.py
cd ..

echo "All done. "
