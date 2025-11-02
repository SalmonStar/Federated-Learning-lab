conda create -n tff_env python=3.9 -y
conda activate tff_env

pip install tensorflow==2.14.1
pip install tensorflow-federated==0.86.0
pip install tensorflow-privacy==0.9.0
