# What to do over SSH
If the connection had stayed stable and VS Code successfully opened the remote Linux window inside Google Colab, you would effectively have a fresh "Ubuntu Linux" computer at your fingertips, backed by a Google GPU. 

Because Colab is a fresh environment every time, it doesn't have your project files or the correct Python version (Python 3.6 for TensorFlow 1.12) installed by default. 

Here is exactly what you would have done in the VS Code terminal (which would now be acting as the Colab Terminal):

### 1. Transfer Your Files
You would drag and drop your entire `Omnia_Anomaly_Detection_coreX` folder from your Windows desktop directly into the left-hand sidebar of VS Code (the Explorer tab). This uploads the files directly to the Colab server over SSH.
*Alternatively, if your code was on GitHub, you would type `git clone <your-repo>` into the terminal.*

### 2. Enter Your Project Folder
```bash
cd Omnia_Anomaly_Detection_coreX
```

### 3. Install Miniconda (To get Python 3.6)
Colab uses Python 3.10 natively, which breaks your older TensorFlow code. You would copy and paste these lines into the terminal to install a fresh Conda environment manager:
```bash
# Download and install Miniconda silently
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda

# Activate conda
source $HOME/miniconda/bin/activate

# Create and activate a pristine Python 3.6 environment
conda create --name corex_env python=3.6 -y
conda activate corex_env
```

### 4. Install the Required CUDA Toolkit (For the GPU)
TensorFlow 1.12 specifically requires an older piece of Nvidia software (CUDA 9.0) to talk to the GPU. You would run this to install it inside your Conda environment:
```bash
conda install cudatoolkit=9.0 cudnn=7.1.2 -c conda-forge -y
```

### 5. Install Python Dependencies
Next, you would install the required libraries like `tfsnippet`, `zhusuan`, and your `requirements.txt`:
```bash
pip install git+https://github.com/haowen-xu/tfsnippet.git@v0.2.0-alpha1
pip install git+https://github.com/thu-ml/zhusuan.git
pip install -r requirements.txt
```

### 6. Run Your Training Script!
Finally, you would run your main Python script. Because you are connected over SSH directly to Colab, this would run natively using their T4 GPU while streaming the output directly to your VS Code screen in real-time.
```bash
python main.py
```

### 7. Download Your Results
When the training finishes (say, after 100 epochs), the model saves checkpoints to the `model_coreX_v1` and `results` folders. Before you disconnect from the SSH session, you would right-click those folders in the VS Code file explorer and click **"Download"** to bring the trained AI models safely back to your Windows computer.
