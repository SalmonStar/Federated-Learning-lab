# TensorFlow Federated (TFF)
> Status: In Progress

## Environment Setup

### Python Version Requirement
TFF currently supports **Python 3.9.x ~ 3.10.x**  
> ⚠️ Python 3.11+ may incur compatibility issue

You can check your Python version via:
```bash
python3 --version
```

If your system Python version < 3.9, please install Python 3.10.
```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.10 python3.10-dev python3.10-venv python3.10-distutils
```

(Optional) Set Python 3.10 as default (only if needed):
```bash
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2
sudo update-alternatives --config python3
```

### Virtual Environment (Ubuntu)
If your system does not have `venv`, install:
```bash
sudo apt update
sudo apt install -y python3.10-venv
```

Create virtual environment:
```bash
python3.10 -m venv ~/tff-venv # create venv
source ~/tff-venv/bin/activate # activate venv
```

Update pip and basic tools:
```bash
pip install --upgrade pip setuptools wheel
```

Exit the virtual environment:
```bash
deactivate
```

### Install TensorFlow & TensorFlow Federated
```bash
pip install "tensorflow-cpu==2.14.*"
pip install "tensorflow-federated==0.86.0"
pip install tensorflow-privacy==0.9.0
```



