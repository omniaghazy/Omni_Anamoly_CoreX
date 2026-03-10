# How to Connect VS Code Directly to Google Colab GPU

Since your project requires TensorFlow 1.12 (Python 3.6) and you want a smooth development experience directly from your code editor, you can connect your local **VS Code** to a **Google Colab GPU** instance using an extension.

This allows you to edit files in VS Code exactly as you normally do, but when you run `python main.py` in the VS Code terminal, it runs on Colab's powerful GPU instead of your local machine.

---

### Step 1: Install the VS Code Extension
1. Open **VS Code**.
2. Go to the **Extensions** tab (or press `Ctrl+Shift+X`).
3. Search for: **`Remote - Tunnels`** (by Microsoft) and install it.
   *(Alternatively, you can search for community extensions like `ColabCode` or `colab-ssh`, but Cloudflare Tunnels via colab is currently the most stable).*

### Step 2: Prepare the Colab Notebook
We need to run a small piece of code inside a Colab Notebook that essentially opens a "backdoor" tunnel into the Colab server for your VS Code to connect to.

1. Go to [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
2. Change the Runtime to GPU: **Runtime -> Change runtime type -> Hardware accelerator: T4 GPU -> Save**.
3. Copy and paste the following Python code into the first cell of the notebook and **run it**:

```python
# Install the highly popular colab-ssh package
!pip install colab-ssh --upgrade

from colab_ssh import launch_ssh_cloudflared

# Launch the tunnel! (Replace 'YOUR_PASSWORD' with any password you want)
launch_ssh_cloudflared(password="YOUR_PASSWORD")
```

4. The output of that cell will show you exactly what to copy next (it will give you a Hostname, Username, and Password). Leave this Colab tab OPEN in your browser.

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
