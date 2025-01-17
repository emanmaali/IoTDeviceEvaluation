# IoT Device Identification Models

This directory contains the implementation of machine learning models used for evaluating IoT device identification. The models are categorised into three distinct folds: original code, Docker-based code, and rewritten code from the original paper.

## Overview

The models in this directory were designed to evaluate the practicality and effectiveness of machine learning techniques in identifying IoT devices across different network and operational conditions. These implementations facilitate reproducibility and further exploration.

## Directory Structure

```plaintext
Models/
├── Meid20.py                 # Meid20 paper code.
├── Okui22.py                 # Okui22 paper code.
├── Ortiz19.py                # Ortiz19 paper code.
├── Perd20.py                 # Perd20 paper code.
├── Pinh19.py                 # Pinh19 paper code.
├── Siva18.py                 # Siva18 paper code.
├── Yang19.py                 # Yang19 paper code. 
├── requirements.txt          # Required Python libraries
├── license                   # Licenseing 
├── README.md                 # Documentation for the directory
```

## Code Folds

### 1. **Original Code**
The following are the original implementations from prior research papers. They serve as the baseline for evaluation in this project. 
- **[Ahmed22](https://github.com/dilawer11/iot-device-fingerprinting)**: [Analyzing the Feasibility and Generalizability of Fingerprinting Internet of Things Devices.](https://petsymposium.org/popets/2022/popets-2022-0057.php)

### 2. **Docker-Based Code**
These scripts are adapted to run within a Docker environment to ensure ease of deployment and reproducibility:
- **[Dong20](https://github.com/KiteFlyKid/Your-Smart-Home-Can-t-Keep-a-Secret-Towards-Automated-Fingerprinting-of-IoT-Traffic-with-Neural-Net.git)**: [Your smart home can't keep a secret: Towards automated fingerprinting of IoT traffic.](https://dl.acm.org/doi/abs/10.1145/3320269.3384732)
- **[Fan22](https://github.com/AliceAndBobCandy/AutoIoT.git)**: [AutoIoT: Automatically Updated IoT Device Identification With Semi-Supervised Learning.](https://ieeexplore.ieee.org/abstract/document/9795895)

### 3. **Rewritten Code**
The following implementations are rewritten versions of the original code. Note that you may need to adjust the code based on your testing case, the path to your training data, and ensure that features are used in the correct extracted format as outlined in the original paper.
- **[Meid20](https://github.com/emanmaali/IoTDeviceEvaluation/blob/98dd3f53f2aa3e215c6fe12d8ec0049debed75aa/Models/Meid20.py)**: [A novel approach for detecting vulnerable IoT devices connected behind a home NAT.](https://www.sciencedirect.com/science/article/pii/S0167404820302418)
- **[Okui22](https://github.com/emanmaali/IoTDeviceEvaluation/blob/98dd3f53f2aa3e215c6fe12d8ec0049debed75aa/Models/Okui22.py)**: [Identification of an IoT device model in the home domain using IPFIX records.](https://ieeexplore.ieee.org/abstract/document/9842469)
- **[Siva18](https://github.com/emanmaali/IoTDeviceEvaluation/blob/98dd3f53f2aa3e215c6fe12d8ec0049debed75aa/Models/Siva18.py)**: [Classifying IoT Devices in Smart Environments Using Network Traffic Characteristics.](https://ieeexplore.ieee.org/abstract/document/8440758)
- **[Pinh19](https://github.com/emanmaali/IoTDeviceEvaluation/blob/98dd3f53f2aa3e215c6fe12d8ec0049debed75aa/Models/Pinh19.py)**: [Identifying IoT devices and events based on packet length from encrypted traffic.](https://www.sciencedirect.com/science/article/abs/pii/S0140366419300052)
- **[Yang19](https://github.com/emanmaali/IoTDeviceEvaluation/blob/98dd3f53f2aa3e215c6fe12d8ec0049debed75aa/Models/Yang19.py)**: [Towards automatic fingerprinting of IoT devices in the cyberspace.](https://www.sciencedirect.com/science/article/abs/pii/S1389128618306856)
- **[Ortiz19](https://github.com/emanmaali/IoTDeviceEvaluation/blob/98dd3f53f2aa3e215c6fe12d8ec0049debed75aa/Models/Ortiz19.py)**: [DeviceMien: Network device behavior modeling for identifying unknown IoT devices.](https://dl.acm.org/doi/abs/10.1145/3302505.3310073)
- **[Perd20](https://github.com/emanmaali/IoTDeviceEvaluation/blob/98dd3f53f2aa3e215c6fe12d8ec0049debed75aa/Models/Perd20.py)**: [IoTfinder: Efficient large-scale identification of IoT devices via passive DNS traffic analysis.](https://ieeexplore.ieee.org/abstract/document/9230403)
