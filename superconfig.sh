# 此代码是用于申请了阿里云的实例后，为了进行训练网络而进行的环境安装过程，系统为 Ubuntu18.04,实例有GPU支持

sudo apt-get update
sudo apt update
sudo apt-get install -y openssl libssl-dev libsqlite3-dev libxml2 libxml2-dev  python-libxslt1 libxslt1-dev python3-pyasn1  libffi-dev  python-pyasn1-modules

# 安装python
cd ~
wget http://www.python.org/ftp/python/3.6.7/Python-3.6.7.tgz # 可以先自行下载然后拷贝到 ~ 目录下，即可跳过此行代码
tar -xvzf Python-3.6.7.tgz
cd Python-3.6.7
./configure --with-ssl
make
sudo make install
python --version
cd /usr/bin
rm python
sudo ln -s python3 python
python --version

# 安装pip&git
sudo apt-get -y install python3-pip
cd /usr/local/bin
rm pip
sudo ln -s pip3 pip
pip install --upgrade pip

sudo apt-get -y install git

# 安装所需要的包
pip install numpy scipy matplotlib pandas tkinter

# torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl 包需要先下载然后拷贝到 ~ 目录下
cd ~
pip install ./torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip install torchvision

# 安装cuda10，cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb 包需要先下载然后拷贝到 ~ 目录下
cd ~
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

# 安装cudnn，cudnn-10.0-linux-x64-v7.4.2.24.tgz 包需要先下载然后拷贝到 ~ 目录下
cd ~
tar -xvzf cudnn-10.0-linux-x64-v7.4.2.24.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*

# 其他依赖
sudo apt-get install python3.6-tk

# 安裝face_recognition
sudo apt -y install cmake

cd ~
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build; cd build; cmake ..; cmake --build .
cd ..
python3 setup.py install

pip install face_recognition

