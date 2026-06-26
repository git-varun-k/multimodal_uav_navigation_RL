# Autonomous Drone Navigation using Reinforcement Learning and Multimodal Feature Fusion

## Overview

This project presents an autonomous drone navigation framework designed for operation in GPS-denied and obstacle-rich environments. The approach combines reinforcement learning with multimodal feature encoding to enable decision-making using RGB and depth information.

Traditional drone navigation systems often depend on GPS signals, hand-crafted control rules, or expensive real-world testing. In challenging environments such as industrial facilities, subterranean spaces, or environments with limited visibility, these approaches become unreliable.

This project addresses those challenges by learning navigation behavior directly from visual observations and depth sensing.

---

## Motivation

The growing adoption of UAVs in industries, emergency response, autonomous inspection, and tactical navigation creates demand for intelligent onboard control systems.

However, autonomous navigation faces several challenges:

- Limited or unavailable GPS signals
- High-dimensional sensor data
- Collision risks in dense environments
- Sim-to-real performance gaps
- Computational constraints during real-time flight

To improve robustness, this project introduces a latent representation approach that compresses RGB and depth observations into a compact feature space for reinforcement learning.

---

## Methodology

### 1. Multimodal Feature Encoding

A custom **FusionEncoder** processes:

- RGB images
- Depth maps

The encoder transforms sensor inputs into a compact latent representation.

```text
RGB + Depth
     │
     ▼
FusionEncoder
     │
     ▼
128-D Latent Vector
     │
     ▼
PPO Policy
     │
     ▼
Navigation Action
```

---

### 2. Reinforcement Learning

Navigation is learned using **Proximal Policy Optimization (PPO)** implemented with Stable-Baselines3.

**Action Space**

- `0` → TURN
- `1` → MOVE STRAIGHT

**Reward Shaping**

- Safe obstacle avoidance
- Forward motion in open regions
- Exploration with minimal collisions

---

### 3. Dataset

Training and evaluation use the **TartanAir Dataset**.

**Training Pipeline**

```text
Office Environment
        │
        ▼
Train PPO Policy
        │
        ▼
Factory Environment
        │
        ▼
Domain Adaptation
        │
        ▼
Evaluation
```

**Dataset Structure**

```text
data/
├── office/
│   └── Easy/
│       └── P000/
│           ├── image_left/
│           └── depth_left/
└── abandonedfactory/
    ├── Easy/
    │   ├── P002/
    │   ├── P004/
    │   └── P006/
```

---

## Training

### Train Vision Encoder

```bash
python train.py
```

Output:

```text
fusion_encoder_v1.pth
```

### Train Reinforcement Learning Agent

```bash
python drone_agent.py
```

Outputs:

```text
drone_pilot_v1.zip
drone_pilot_adapted_factory.zip
```

---

## Evaluation

Run the evaluation script:

```bash
python evaluate_final.py
```

Evaluation metrics include:

- Total Reward
- Average Reward
- Navigation Accuracy
- Blocked Frame Detection

---

## Technologies

- Python
- PyTorch
- Stable-Baselines3
- Gymnasium
- NumPy
- Torchvision
- Reinforcement Learning
- Proximal Policy Optimization (PPO)

---

## Future Work

- Continuous control actions
- Temporal memory architectures
- Sim-to-real transfer
- Real drone deployment
- Improved reward shaping

---

## Repository Structure

```text
uav_navigation_RL/
├── source/
├── figures/
├── csv/
├── requirements.txt
└── README.md
```

---

## Author

**Varun Kumar Kolloju**

MS Data Science Project

**Autonomous UAV Navigation using Reinforcement Learning and Multimodal Sensor Fusion**