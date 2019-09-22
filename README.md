# MIDS Capstone
Repository for best MIDS Capstone ever!

## Importing Video and Processing on Cloud instance
Follow these steps to download video (if it isn't already downloaded) and to process it with opencv. **NOTE:** future state we will not need to install things everytime, but for now I don't want to keep a VM running at all times, so we need to install every time

create a VM and ssh into your VM

```sh
# For now Alex will create VMs, we will get a static ip once we decide on AWS or
# IBM cloud
ssh root@ip_of_your_vm
```

install a bunch of packages

```sh
apt update
apt install -y git python3-opencv python3-pip libopencv-dev
```

wait 10-15 minutes for this to finish :face_with_rolling_eyes:. After installation, git clone the project repo

```sh
git clone https://github.com/ahsenq/bittah-ninja.git
```

pip install virtualenv and create an environment

```sh
pip3 install virtualenv
virtualenv w210
source w210/bin/activate
```

install the requirements for the environment

```sh
cd bittah-ninja
pip3 install -r requirements.txt
```

download the videos. You can either just run the download script, or run interactively from a jupyter kernel

```sh
# option 1
python3 importFromIBM.py
# option 2
jupyter lab --ip=0.0.0.0 --allow-root
# then paste the link into your browser and change the ip to the ip of the VM
# then open the importFromIBM.ipynb notebook
```

once the videos are downloaded, the data is ready to explore. You can play around with the readVideo notebook which has some starter code for reading in a video and converting it to a numpy array
