# DALFE: A Dual-aligned Framework via LLM-based Adaptive Feature Extraction for Time Series Forecasting

## Description
Time series forecasting plays a vital role in IoT systems. The recent introduction of Large Language Model (LLM) into time series forecasting has somewhat addressed the shortcomings of existing statistical and deep learning methods in interpreting the semantic features of time series data. 
However, existing methods that guide LLM to understand data and generate features through time series-based prompts, followed by simple concatenation or alignment of these features, lack a thorough exploration of fine-grained knowledge. To address this issue, we propose a Dual-aligned framework via LLM-based Adaptive Feature Extraction for time series forecasting (DALFE). 
This framework adaptively selects the most suitable feature layers from transformer-based LLMs and employs a cross-model feature alignment module to achieve efficient integration and consistency adjustment between time series features and LLM semantic information. DALFE employs a transformer architecture to encode both the time series data and the prompt information, and then utilizes a decoder to predict future states. The effectiveness of DALFE has been validated across multiple public real-world datasets, with experimental results indicating that DALFE exhibits exceptional predictive performance.

<img width="612" alt="fdb4749df093ae7617054380d6eab9e" src="https://github.com/user-attachments/assets/bc696e24-e7f3-40b2-bef2-dd59b2bb9598" />


## Main Features
- Supports multiple time series datasets (such as ETT, ILI, etc.)
- Provides custom dataset loaders
- Implements cross-modal alignment models based on Transformer
- Includes data preprocessing and feature engineering tools
- Offers complete training and evaluation workflows

## Installation & Usage
1. Clone the repository to your local machine
2. Create conda environment:
   ```bash
   conda env create -f env_ubuntu.yaml
   ```
3. Activate the environment:
   ```bash
   conda activate your_env_name
   ```
4. Prepare the dataset: TimesNet
5. Run the training script:
   ```bash
   bash scripts/Store_ETTh1.sh
   bash scripts/ETTh1.sh
   ```

