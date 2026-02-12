# energy-aware-live-transcoding-drl

Implementation of a DRL-based bitrate selection and budget allocation framework for live video transcoding systems.

This repository contains training and decision (inference) code for a PPO-based model.  
The trained model is saved as a `.zip` file and later used during the decision phase.

---

## Overview

This repository is organized into three main components:

1. Model training
2. Environment and algorithm definitions
3. Decision and evaluation

The main execution entry point is `train.py`.

---

## File Structure

- `train.py`
  - Training entry point
  - Initializes environment and hyperparameters
  - Runs PPO training
  - Saves trained model as a `.zip` file

- `knapsack.py`
  - Core implementation module
  - Defines the environment
  - Implements state representation and reward computation
  - Contains the `step()` transition function
  - Includes PPO-related definitions
  - Implements baseline / comparison algorithms

- `training.py`
  - Loads a trained `.zip` model
  - Performs decision / inference
  - Evaluates model performance

---

## Execution

```bash
python train.py

Output:

Trained model saved as a .zip file (e.g., model.zip)

2) Decision / Evaluation Phase

python train.py

This step:

Loads the trained .zip model

Executes bitrate selection and allocation decisions

Evaluates performance
