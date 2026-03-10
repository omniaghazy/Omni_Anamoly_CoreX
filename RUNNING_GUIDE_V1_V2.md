# Omnia CoreX: Running Guide for Version 1 & 2

This guide explains how to properly run Version 1 and Version 2 of the Omnia Anomaly Detection project. Both versions use a TensorFlow 1.15 environment and require specific setup steps depending on whether you are running locally or on Kaggle/Colab.

## 1. Environment Setup (Local Windows)
Both V1 and V2 rely on Python 3.6 and TensorFlow 1.15 to support the underlying OmniAnomaly model.
1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Open Anaconda Prompt and create the environment:
   ```bash
   conda create -n py36 python=3.6
   conda activate py36
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: This will install TensorFlow 1.15.5, Zhusuan, and TFSnippet)*

## 2. Running Version 1 (V1)
Version 1 is located in `X:\Omnia_CoreX\Omnia_Anomaly_Detection_coreX-main\`.
1. **Preprocess the data**:
   Run the preprocessing script to clean `rtde_data.csv` and generate the required `.pkl` files in the `data/` folder.
   ```bash
   python data_preprocess.py
   ```
   *(Alternatively, run the `run_preprocess.bat` file)*
2. **Train & Evaluate the Model**:
   Execute `main.py` to start training.
   ```bash
   python main.py
   ```
   *(Alternatively, run the `run_train.bat` file)*
3. **View Results**:
   Results and plots will be saved in the `results/` folder.

## 3. Running Version 2 (V2)
Version 2 is located in `X:\Omnia_CoreX\Omnia_CoreX_v2\Omnia_Anomaly_Detection_coreX-main\`. It includes enhanced feature engineering and a streamlined pipeline.
1. **Full Pipeline Execution (Recommended)**:
   V2 includes a batch script to run everything sequentially. Run this in your command prompt:
   ```bash
   run_all.bat
   ```
2. **Manual Execution**:
   If you wish to do it step by step:
   - `python data_preprocess.py` (Extracts and cleans raw data)
   - `python feature_engineering.py` (Applies advanced feature extraction)
   - `python main.py` (Trains the model)
   - `python plot_results.py` (Generates visual evaluation metrics)

## 4. Running on Kaggle (V2)
Due to substantial computational requirements (e.g. 10h+ processing time for 3 epochs), Kaggle is recommended. We've prepared specific scripts to make deploying onto Kaggle easier.
1. Run `zip_for_kaggle.py` (located in the root directory: `X:\Omnia_CoreX\zip_for_kaggle.py`) to zip the V2 folder securely.
2. Upload the resulting `.zip` as a dataset to your Kaggle Notebook.
3. In Kaggle, use a notebook environment set to **Python** with **GPU (P100 or T4x2)**.
4. Run the custom installation & execution cells we set up in our Kaggle runs to compile a custom Conda Python 3.6 environment on Kaggle's image.
