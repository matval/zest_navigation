# ZeST Navigation
An LLM-based Zero-Shot Traversability Prediction Method

## System requirements
- Linux (Tested on Ubuntu 20.04)
- ROS Noetic
- Python3 (Tested using Python 3.8) 
- CUDA (Tested using CUDA 12.0)
- PyTorch (Tested using Pytorch 2.2.2) 

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
    pip3 install torch==2.2.2 torchvision
    ```

- Install the python requirements.
    ```shell
    pip3 install -r requirements.txt
    ```

## Running the package
To use the traversability package, run
```shell
roslaunch wayfast_zero run_wayfast_zero.launch
```
