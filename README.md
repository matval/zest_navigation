# ZeST Navigation
An LLM-based Zero-Shot Traversability Navigation Method

## System requirements
- Linux (Tested on Ubuntu 20.04)
- ROS Noetic
- Python3 (Tested using Python 3.8) 
- CUDA (Tested using CUDA 12.0)
- PyTorch (Tested using Pytorch 2.1.0)

## Setup
- Install the ROS dependencies. In your ROS workspace, run
  ```shell
  rosdep install --from-paths src --ignore-src -r -y
  ```

- Create a file `api_key.txt` with your OpenAI API key at the root directory of this package.

- Download your preferred SAM model checkpoint.
  - For `vit_b`:
    ```shell
    wget -P llm_prediction/checkpoint https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    ```
  - For `vit_l`:
    ```shell
    wget -P llm_prediction/checkpoint https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
    ```
  - For `vit_h`:
    ```shell
    wget -P llm_prediction/checkpoint https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    ```

- Add the chosen checkpoint to `llm_prediction/config/config/yaml`.

- Create a python virtual environment and activate it.
    ```shell
    python3 -m virtualenv env
    source env/bin/activate
    ```

- Install Pytorch.
    ```shell
    pip3 install torch==2.1.0 torchvision
    ```
- If running on a Jetson device, choose the Pytorch according to the JetPack
    ```shell
    pip3 install https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
    ```
- If running on a Jetson device, install `torchvision` from source
	```shell
	sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
	git clone --branch <version> https://github.com/pytorch/vision torchvision
	cd torchvision
	export BUILD_VERSION=0.x.0  # where 0.x.0 is the torchvision version. For Pytorch 2.1.0, choose 0.16.1  
	/home/companion/catkin_ws/src/zest_navigation/env/bin/python setup.py install
	```
	where `/home/companion/catkin_ws/src/zest_navigation/env/bin/python` is the Python interpreter from your virtualenv.

- Install the python requirements.
    ```shell
    pip3 install -r requirements.txt
    ```

## Running the package
To use the traversability package on your computer, run
```shell
roslaunch llm_prediction run_zest_offline.launch
```

To use the traversability package on the Terrasentia, run
```shell
roslaunch llm_prediction run_zest_terrasentia.launch
```
