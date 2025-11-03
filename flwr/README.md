# Flower + TensorFlow
## Environment
```bash
sudo apt update
sudo apt install python3-pip # if you haven't install pip
pip install flwr tensorflow
```

## Running
Before running the code, make sure that you are in the flwr folder.
```bash
cd Federated-Learning-lab/flwr
```

Server (CloudLab node 1) 
```bash
python3 server.py
```

Client (CloudLab node 2)
```bash
python3 client.py <SERVER_IP:PORT>
```
For example:
```bash
python3 client.py 127.0.0.1:8080
```

## Acknowledgments
The Flower baseline implementation in this directory is based on the official tutorial video:

🎬 *Federated Learning with Flower & TensorFlow*  
https://youtu.be/FGTc2TQq7VM?si=BQNHDaA5sgYRenOw  

Thanks to the Flower team for providing clear instructional material and starter code.
