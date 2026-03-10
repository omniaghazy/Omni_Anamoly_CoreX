# How to Connect VS Code Directly to Google Colab GPU

# 🚀 coreX Colab & VS Code Guide (UPDATED)

The best way to run your project on Colab from VS Code is to use **Google Drive**. This keeps your files synced and automatically saves your results.

---

### Phase 1: Preparation (Local)
1. **Upload your local project folder** (`Omni_Anomaly_Detection_coreX`) to your Google Drive root (`My Drive`).
2. Open your local VS Code.
3. Install the **'Google Colab'** extension from the Marketplace.

---

### Phase 2: Connecting to Colab from VS Code
1. Open the [Run_on_Colab.ipynb](file:///d:/Omni_Anomaly_Detection_coreX/Run_on_Colab.ipynb) file in VS Code.
2. Click the **'Select Kernel'** button (top-right).
3. Select **'Google Colab'** → Sign in to your Google Account.
4. Select the **T4 GPU** runtime.

---

### Phase 3: Running the Notebook
Follow the steps inside the notebook:
- **Step 1 (Drive Mount):** This will ask for permission to access your Drive. Grant it. It will automatically find your project folder.
- **Step 2 (Environment Setup):** This cell will restart your session. Just wait for it to reconnect.
- **Steps 3 to 5:** Run them in order to install dependencies and start the **100-epoch training**.

---

### 💡 Pro Tips
- **Persistent Results:** Because we are using Google Drive, all your training logs and `model_coreX_v1/` checkpoints will be saved **directly to your Drive**. You don't need to manually download them!
- **No Upload Errors:** We removed the `files.upload()` widget which caused crashes in VS Code.
- **Check the GPU:** Always ensure Step 4 confirms `[OK] GPU detected` or your training will be extremely slow.

🚀 **Go to your Google Drive, upload the folder, then try Run_on_Colab.ipynb again!**

### Step 3: Connect VS Code to the Tunnel
Now that Colab has a tunnel open, we tell VS Code to enter it.

1. In VS Code, open the **Command Palette** (`Ctrl+Shift+P`).
2. Type and select: **`Remote-SSH: Connect to Host...`**
3. Select **`+ Add New SSH Host...`**
4. Paste the SSH string that Colab gave you in Step 2. (It will look something like: `ssh root@some-random-url.trycloudflare.com`).
5. It will ask which configuration file to update. Select the default one (usually `C:\Users\YourUser\.ssh\config`).
6. Click **Connect** in the pop-up at the bottom right.
7. It will ask for the password. Enter the password you chose in Step 2 (`YOUR_PASSWORD`).

**Boom! 🚀**
Your VS Code terminal and file explorer are now completely inside the Google Colab server!

### Step 4: Setup the Environment & Run Training
Now that you are inside the Colab server via VS Code, you can see the terminal is running Linux (Colab).

1. Clone or clone your code directly into the Colab environment by opening a terminal in VS Code:
```bash
git clone <your-github-repo-url>
# OR upload your files directly by dragging them into the VS Code file explorer
```

2. Run the environment setup scripts (to install Python 3.6 and TF 1.12):
```bash
# In the VS Code Terminal (connected to Colab)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
source $HOME/miniconda/bin/activate
conda create --name corex_env python=3.6 -y
conda activate corex_env

# Install legacy CUDA
conda install cudatoolkit=9.0 cudnn=7.1.2 -c conda-forge -y

# Install dependencies
pip install git+https://github.com/haowen-xu/tfsnippet.git@v0.2.0-alpha1
pip install git+https://github.com/thu-ml/zhusuan.git
pip install -r requirements.txt
```

3. Train your model!
```bash
python main.py
```

### Important Warning
Google Colab wipes its servers when you disconnect or after ~12 hours of inactivity. **Always download your `/results` and `save_dir` checkpoint folders back to your local machine before closing Colab!** You can do this easily by right-clicking the folder in the VS Code file explorer and selecting **Download**.
