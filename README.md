# IMU Debiasing

## Overview
This project focuses on debiasing (denoising) Inertial Measurement Unit (IMU) data, including gyroscope and accelerometer measurements. A neural network explicitly models the bias dynamics, while a neural ODE on $SO(3)$ is designed for training. The loss is computed using ground truth orientation, velocity, and position, without requiring ground truth for bias.

The overall framework is illustrated below:

<img src="figs/fig1.png" alt="Framework" width="600">


### Paper
Coming soon

## Setup

### Create a Virtual Environment
To ensure an isolated environment for dependencies, run the following commands (Python version: 3.10.12):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r Requirments.txt
```


### Prepare the Dataset
The required IMU data is provided in the `data/` folder. Alternatively, you can download data from the following sources:

- [EUROC](http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/)
- [TUM-VI](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset)

**Note:** For TUM-VI, the IMU data is extracted from the raw-data rosbag, not the calibrated rosbag, but uses the synthesized timestamps from the calibrated rosbag.



## Begin training!
Run the following commands in the terminal to start training (ensure you are using python interpreter from venv/)
```
python3 BiasDy/mainEuroc.py
python3 BiasDy/mainTUM.py
python3 BiasDy/mainFetch.py
```

## Results
The full results will be saved to `./results`. Partial results:

<img src="figs/MH_04_difficult.png" alt="Framework" width="600">
<img src="figs/dataset_room4.png" alt="Framework" width="600">

## Citation
To be added once the corresponding paper is published.



## Acknowledgments

This project incorporates code and ideas from the following sources:

- [denoise-imu-gyro](https://github.com/mbrossar/denoise-imu-gyro) by M. Brossard
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq) by Ricky T. Q. Chen
- [NeuralCDE](https://github.com/patrick-kidger/NeuralCDE) by Patrick Kidger



