# Artificial Intelligence in Medicine — REINFORCE Notebook

## Description
This project implements a simulation environment and REINFORCE (Policy Gradient) algorithm for optimizing insulin dosing in hyperglycemic patients. The agent observes patient clinical parameters and learns to take dose actions that maximize long-term glucose control.  

Offline RL is used in a simulated environment, ensuring no risk to actual patients.

## Installation

```bash
pip install gymnasium plotly torch matplotlib pandas numpy

This notebook is prepared to run on Google Colab.

Prerequisites: processed_data.npz and scaler.pkl must be generated from Notebook 1 and placed in the same directory.

Contents
Setup and Libraries
Loads PyTorch, Gymnasium, Plotly, and other necessary libraries.
Load Data
Loads preprocessed patient data (processed_data.npz) and prepares arrays such as X, y, states, actions, rewards.
Simulation Environment (MedicalEnv)
Glucose control simulation adapted from CartPole:
State: normalized patient parameters
Action: 0=Decrease | 1=Keep | 2=Increase
Reward: stay in target range, hypoglycemia/hyperglycemia penalties, consecutive dose increase penalty
REINFORCE Agent
Three-layer MLP policy network.
Entropy bonus used for exploration-exploitation balance.
compute_returns and reinforce_update perform policy gradient updates.
Training Loop
Trains for a set number of episodes.
Best model is saved (best_policy.pth).
Rewards, episode length, and loss are visualized during training.
Results Visualization
Uses Matplotlib and Plotly to display episode rewards, step counts, and action distributions interactively.
Simulates a sample episode using the best policy.
Clinical Evaluation
Calculates average reward, median, episode length, and action distribution over test episodes.
Safety-focused action preferences are analyzed.
Saving and Downloading
Saves model and plots into a .zip file for download.
Ethics and Safety
Simulation ensures zero patient risk.
Hypoglycemia and hyperglycemia penalties are carefully tuned.
Consecutive dose increase penalty reduces overdose risk.
Extensive validation is required before real clinical deployment.
Suggested Improvements
More advanced algorithms: PPO, A2C for more stable training.
Include past dose and biological rhythms in the state representation.
Model individual pharmacodynamic variability and more precise clinical risk functions.
File Structure
├─ processed_data.npz              # Preprocessed data
├─ scaler.pkl                       # Normalization scaler
├─ MedicalEnv.ipynb                 # Simulation & training notebook
├─ best_policy.pth                  # Best model checkpoint
├─ reinforce_final.pth              # Final model & optimizer state
├─ results_01_training.png          # Training summary plots
├─ results_02_sample_episode.png    # Sample episode visualization
├─ ISE427_model_outputs.zip         # All outputs zipped
└─ README.md                        # This file
References
OpenAI Gymnasium: https://gymnasium.farama.org/
REINFORCE Policy Gradient: Sutton & Barto, Reinforcement Learning (2nd Edition)
Kaggle glucose datasets and clinical simulation literature
