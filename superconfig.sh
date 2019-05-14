# �˴��������������˰����Ƶ�ʵ����Ϊ�˽���ѵ����������еĻ�����װ���̣�ϵͳΪ Ubuntu18.04,ʵ����GPU֧��

sudo apt-get update
sudo apt update
sudo apt-get install -y openssl libssl-dev libsqlite3-dev libxml2 libxml2-dev  python-libxslt1 libxslt1-dev python3-pyasn1  libffi-dev  python-pyasn1-modules

# ��װpython
cd ~
wget http://www.python.org/ftp/python/3.6.7/Python-3.6.7.tgz # ��������������Ȼ�󿽱��� ~ Ŀ¼�£������������д���
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

# ��װpip&git
sudo apt-get -y install python3-pip
cd /usr/local/bin
rm pip
sudo ln -s pip3 pip
pip install --upgrade pip

sudo apt-get -y install git

# ��װ����Ҫ�İ�
pip install numpy scipy matplotlib pandas tkinter

# torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl ����Ҫ������Ȼ�󿽱��� ~ Ŀ¼��
cd ~
pip install ./torch-1.0.1.post2-cp36-cp36m-linux_x86_64.whl
pip install torchvision

# ��װcuda10��cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb ����Ҫ������Ȼ�󿽱��� ~ Ŀ¼��
cd ~
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

# ��װcudnn��cudnn-10.0-linux-x64-v7.4.2.24.tgz ����Ҫ������Ȼ�󿽱��� ~ Ŀ¼��
cd ~
tar -xvzf cudnn-10.0-linux-x64-v7.4.2.24.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*

# ��������
sudo apt-get install python3.6-tk

# ���bface_recognition
sudo apt -y install cmake

cd ~
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build; cd build; cmake ..; cmake --build .
cd ..
python3 setup.py install

pip install face_recognition

