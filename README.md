# ZeST Navigation
An LLM-based Zero-Shot Traversability Navigation Method

## ⚙️ Environment Setup

ZeST runs in a **ROS Noetic catkin workspace**. The original reference environment is Ubuntu 20.04, Python 3.8, PyTorch 2.1.0, and CUDA 12.0.

**Required:** an NVIDIA GPU with a CUDA-enabled PyTorch build. MPPI currently uses CUDA directly; CPU-only PyTorch cannot run the controller.

### ROS dependencies

Install [ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu) on Ubuntu 20.04 first. Noetic is end-of-life; initialize rosdep once and explicitly include its archived distribution:

```bash
sudo apt-get install build-essential cmake python3-pip python3-rosdep python3-virtualenv
# Run only if rosdep has not already been initialized:
sudo rosdep init
rosdep update --rosdistro noetic --include-eol-distros
```

Place this repository in your catkin workspace's `src/` directory. From the workspace root, run:

```bash
source /opt/ros/noetic/setup.bash
rosdep install --from-paths src --ignore-src --rosdistro noetic -r -y
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
```

### Python environment

From the repository root, create and activate the `env/` environment expected by the launch files. Use Noetic's system Python 3.8 and system-site packages so ROS modules and compiled Python bindings remain compatible:

```bash
python3 -m pip install virtualenv
python3 -m virtualenv --python=/usr/bin/python3 --system-site-packages env
source env/bin/activate
```

### PyTorch

Choose **one** installation path for your platform.

#### Desktop

Install PyTorch 2.1.0 for the MPPI controller:

```bash
python -m pip install "numpy>=1.19.1,<2"
python -m pip install torch==2.1.0
```

See the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/) for CUDA-specific wheels.

#### NVIDIA Jetson

Use an NVIDIA PyTorch build matching your **JetPack release, Python version, and architecture**. The following wheel is the original installation example for **JetPack 5.1.2, Python 3.8, and ARM64**:

```bash
sudo apt-get install libopenblas-dev
python -m pip install "numpy>=1.19.1,<2"
python -m pip install https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
```

For another JetPack release, select the corresponding wheel from [NVIDIA's PyTorch for Jetson instructions](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) instead.

### Python requirements

Finally, from the repository root with `env/` still activated:

```bash
python -m pip install -r requirements.txt
```

Create `api_key.txt` in the repository root containing your OpenAI API key.

## Running the package
To use the traversability package on your computer, run
```shell
roslaunch llm_prediction run_zest_offline.launch
```

To use the traversability package on the Terrasentia, run
```shell
roslaunch llm_prediction run_zest_terrasentia.launch
```
