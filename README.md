# Reinforcement Learning stablebaselines3
 

![portfolio-9](https://github.com/shub-garg/Reinforcement-Learning-stablebaselines3/assets/52582943/85a058e5-01f5-42c6-80e2-550938c8d7bf)

| **Name**         | **Recurrent**      | `Box`          | `Discrete`     | `MultiDiscrete` | `MultiBinary`  | **Multi Processing**              |
| ------------------- | ------------------ | ------------------ | ------------------ | ------------------- | ------------------ | --------------------------------- |
| ARS   | :x: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :heavy_check_mark: |
| A2C   | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| DDPG  | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |
| DQN   | :x: | :x: | :heavy_check_mark: | :x:                 | :x:                | :heavy_check_mark: |
| HER   | :x: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :heavy_check_mark: |
| PPO   | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| QR-DQN  | :x: | :x: | :heavy_check_mark: | :x:                 | :x:                | :heavy_check_mark: |
| RecurrentPPO   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| SAC   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |
| TD3   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |
| TQC   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x: | :heavy_check_mark: |
| TRPO  | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| Maskable PPO  | :x: | :x: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:  |


# Custom CNN Policy with ResNet-like Structure for CarRacing-v2

## Description

The custom policy is built using a convolutional neural network (CNN) with a ResNet-like architecture. ResNet (Residual Network) structures are effective for training deep neural networks by allowing gradients to flow through the network directly, avoiding the vanishing gradient problem. This custom CNN policy is designed to extract features from the CarRacing-v2 environment's observation space, which consists of image data.

## Architecture

### ResNet Block
The ResNet block is the fundamental building block of the network. It consists of two convolutional layers with batch normalization and ReLU activations. A skip connection (shortcut) is added to bypass the block, which helps in preserving the gradient flow.

### Custom CNN
The custom CNN is composed of a series of ResNet blocks followed by an adaptive average pooling layer and a linear layer for feature extraction.

1. **Input:** The observation space from the CarRacing-v2 environment.
2. **ResNet Block 1:** 32 filters, stride 1
3. **ResNet Block 2:** 64 filters, stride 2
4. **ResNet Block 3:** 128 filters, stride 2
5. **ResNet Block 4:** 256 filters, stride 2
6. **Adaptive Average Pooling:** Reduces each channel to a single value.
7. **Linear Layer:** Fully connected layer to produce the final feature vector.

## Diagram

```plaintext
Input Image
     |
ResNet Block 1 (32 filters, stride 1)
     |
ResNet Block 2 (64 filters, stride 2)
     |
ResNet Block 3 (128 filters, stride 2)
     |
ResNet Block 4 (256 filters, stride 2)
     |
Adaptive Average Pooling
     |
Fully Connected Layer (256 units)
     |
Feature Vector
