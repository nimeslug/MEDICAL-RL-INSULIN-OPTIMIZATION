# Artificial Intelligence in Medicine

## Reinforcement Learning for Insulin Dosage Decision Support

This project explores the use of **Reinforcement Learning (RL)** techniques to support **insulin dosage decision-making** for diabetes patients using clinical data.

The system builds a **simulation environment representing a patient's glucose balance** and trains an RL agent using the **REINFORCE policy gradient algorithm** to learn optimal dosage adjustment strategies.

The project pipeline consists of two main stages:

1. **Data Acquisition, Exploratory Data Analysis (EDA), and Preprocessing**
2. **Simulation Environment and Reinforcement Learning Training**

The workflow transforms raw clinical data into an RL-compatible dataset and trains a policy network that learns insulin dosage adjustment decisions.

---

# Project Workflow

```
Raw Clinical Dataset (Kaggle)
            │
            ▼
Notebook 1
Data Acquisition + EDA + Preprocessing
            │
            ▼
processed_data.npz
scaler.pkl
            │
            ▼
Notebook 2
Medical Simulation Environment
+ REINFORCE Training
            │
            ▼
Trained Policy Model
Training Metrics
Simulation Results
```

---

# Project Structure

```
Artificial-Intelligence-in-Medicine/

│
├── TYZ_Data_EDA_Preprocessing.ipynb
├── TYZ_Model_Training.ipynb
│
├── processed_data.npz
├── scaler.pkl
│
├── best_policy.pth
├── reinforce_final.pth
│
├── eda_01_missing_values.png
├── eda_02_target_distribution.png
├── eda_03_distributions.png
├── eda_04_correlation.png
├── eda_05_outliers.png
├── eda_06_violin_by_target.png
├── eda_07_preprocessing_comparison.png
│
├── results_01_training.png
├── results_02_sample_episode.png
│
└── README.md
```

---

# Dataset

Dataset used in this project:

**Kaggle — Diabetes Clinical Dataset**

https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset

The dataset contains clinical measurements related to diabetes management and insulin treatment decisions.

The goal is to learn a policy that selects the correct **insulin dosage adjustment action**.

Possible decisions:

| Action | Meaning                  |
| ------ | ------------------------ |
| 0      | Decrease insulin         |
| 1      | Keep insulin dose stable |
| 2      | Increase insulin         |

---

# Notebook 1 — Data Acquisition, EDA and Preprocessing

Notebook 1 performs the **data preparation pipeline** required before training the reinforcement learning agent.

### Steps

### 1. Data Acquisition

The dataset is downloaded directly from Kaggle using the Kaggle API.

### 2. Exploratory Data Analysis (EDA)

Several analyses are performed:

* Missing value analysis
* Target variable distribution
* Feature distributions
* Correlation analysis
* Outlier detection (IQR method)
* Feature distribution by decision class

### Generated Visualizations

* Missing value ratio
* Target class distribution
* Feature histograms
* Correlation heatmap
* Boxplot-based outlier detection
* Violin plots by target class
* Before/after preprocessing comparison

### 3. Data Preprocessing

The preprocessing pipeline includes:

| Step                   | Method            |
| ---------------------- | ----------------- |
| Categorical Encoding   | LabelEncoder      |
| Missing Value Handling | Median Imputation |
| Outlier Handling       | IQR Clipping      |
| Feature Scaling        | StandardScaler    |

### 4. Reinforcement Learning Dataset Preparation

Clinical samples are transformed into RL trajectories:

```
(state, action, reward, next_state)
```

Reward design:

| Action        | Reward |
| ------------- | ------ |
| Keep dose     | +1.0   |
| Decrease dose | -1.0   |
| Increase dose | -2.0   |

Outputs generated:

```
processed_data.npz
scaler.pkl
```

These files are used by **Notebook 2**.

---

# Notebook 2 — Simulation Environment and RL Training

Notebook 2 trains a reinforcement learning agent using the **REINFORCE Policy Gradient algorithm**.

### Key Components

* Custom medical simulation environment
* Policy neural network
* Reinforcement learning training loop
* Evaluation and visualization tools

---

# Medical Simulation Environment

A custom **OpenAI Gymnasium environment** called `MedicalEnv` is implemented.

The environment simulates insulin dosage control.

Conceptual analogy:

| CartPole Concept | Medical Concept      |
| ---------------- | -------------------- |
| Pole angle       | Glucose deviation    |
| Balance          | Stable glucose level |
| Pole fall        | Severe hyperglycemia |

Environment characteristics:

* **State**: normalized clinical features
* **Action space**: 3 insulin adjustment actions
* **Reward function**: clinical safety-driven

Additional penalty:

```
Repeated dose increase → extra penalty
```

This discourages unsafe insulin escalation.

---

# Reinforcement Learning Algorithm

The agent is trained using the **REINFORCE Policy Gradient method**.

### Policy Network Architecture

```
Input Layer      : Clinical state features
Hidden Layer 1   : 128 neurons + ReLU
Hidden Layer 2   : 128 neurons + ReLU
Dropout          : 0.2
Output Layer     : Softmax action probabilities
```

Training settings:

| Parameter           | Value |
| ------------------- | ----- |
| Episodes            | 500   |
| Discount factor (γ) | 0.99  |
| Optimizer           | Adam  |
| Learning Rate       | 1e-3  |
| Entropy bonus       | 0.05  |

Gradient clipping is applied to improve stability.

---

# Training Outputs

During training the following metrics are tracked:

* Episode rewards
* Episode length
* Policy loss
* Action distribution

Visualization outputs:

```
results_01_training.png
results_02_sample_episode.png
```

Plots include:

* Reward progression
* Policy loss
* Stability duration
* Action distribution

Interactive visualizations are generated with **Plotly**.

---

# Example Simulation

After training, the best policy is evaluated in a simulated episode.

The visualization shows:

* Step-by-step rewards
* Chosen insulin adjustment
* Policy probability evolution

---

# Evaluation Metrics

Clinical evaluation metrics include:

| Metric              | Description                   |
| ------------------- | ----------------------------- |
| Average Reward      | Overall policy performance    |
| Episode Length      | Glucose stability duration    |
| Action Distribution | Policy behavior               |
| Reward Variance     | Stability of learned strategy |

---

# Technologies Used

| Category       | Tools                       |
| -------------- | --------------------------- |
| Programming    | Python                      |
| ML Framework   | PyTorch                     |
| RL Environment | Gymnasium                   |
| Data Analysis  | Pandas, NumPy               |
| Visualization  | Matplotlib, Plotly, Seaborn |
| ML Utilities   | Scikit-learn                |

---

# Installation

Install required libraries:

```
pip install kaggle seaborn plotly scikit-learn gymnasium torch
```

---

# Running the Project

### Step 1 — Run Notebook 1

This will:

* download dataset
* perform EDA
* preprocess data
* generate `processed_data.npz`

### Step 2 — Run Notebook 2

Notebook 2:

* loads processed dataset
* creates simulation environment
* trains the RL agent
* produces model outputs and visualizations

---

# Ethical Considerations

This work is **research-oriented** and does not provide medical advice.

Important considerations:

* Training occurs only in a **simulated environment**
* No real patient treatment decisions are made
* Clinical deployment would require extensive validation and regulatory approval

---

# Limitations

* Real glucose–insulin dynamics are more complex
* Individual patient variability is not fully modeled
* Kaggle dataset may not fully represent clinical populations

Future work may include:

* patient-specific policies
* advanced RL algorithms (PPO, Actor-Critic)
* physiological simulators

---

# Author

Artificial Intelligence in Medicine Project
