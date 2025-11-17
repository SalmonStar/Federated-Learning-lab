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

## TEE Environment and Federated Learning Implementation Attempts

### Initial Strategy: Building a FL Transformation inside Google’s TEE Pipeline

The original plan was to construct a **Federated Learning (FL) transformation** within the Google Confidential Computing (TEE) framework.  
The idea was to:

1. Keep the existing confidential-compute pipeline (attestation, KMS, ledger, sessions).
2. Replace the `fed_sql` transformation in Google’s `confidential-federated-compute` repository.
3. Remove SQL/DP behaviors and implement a lightweight FL-style aggregation (e.g., FedAvg).
4. Run the entire FL workflow inside a TEE container.

-> Modifying the transformation layer under: `confidential-federated-compute/containers/fed_sql`

#### Completed:
- `confidential_transform_server.h`

#### In Progress:
- `confidential_transform_server.cc`  
  - Encountered unresolved dependencies → required further changes
- `ledger_session.cc`
- `ledger_session.h`
- `kms_session.cc`
- `kms_session.h`

#### Why This Approach Was Paused
`fed_sql` is not a full Federated Learning transformation, but it internally
combines SQLite-based per-client processing with federated-style aggregation
steps based on Google’s Aggregation Core. These accumulate/merge/report
mechanisms are the same primitives used in federated computation.

Because SQL logic and aggregation logic are tightly coupled inside the
transformation, removing the SQL parts or repurposing it into a pure FL
transformation leads to dependency conflicts, incompatible input formats, and
breakage across multiple internal modules.



## Acknowledgments
Environment compatibility reference for TFF and TF inspired by:  
📎 https://hackmd.io/@thc1006/SJx5Ve1Iee  

Thanks to the original author for summarizing version constraints clearly — saved lots of debugging time🔥


