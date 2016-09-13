# Installing the dependencies of keras-finetuning on a fresh OS X system.
#
# This is not a proper standalone installation script,
# rather, the idea is that you copy it line by line into a terminal.

# If you already have brew, pip, virtualenv, skip accordingly:
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
sudo easy_install pip
sudo pip install virtualenv

virtualenv venv
. venv/bin/activate

pip install numpy
pip install scipy > cout 2> cerr
pip install theano > cout 2> cerr
brew tap homebrew/science
brew install opencv

# This sets up the earliest version if there are more installed:
PYOPENCV_DIR=(/usr/local/Cellar/opencv/*/lib/python2.7/site-packages)
export PYTHONPATH=${PYOPENCV_DIR[0]}:$PYTHONPATH

brew install hdf5
pip install h5py
pip install pillow

git clone https://github.com/fchollet/keras.git
cd keras/
python setup.py install
cd ..
