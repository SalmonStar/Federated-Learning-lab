# Federated Learning Framework with TensorFlow

This repository compares two FL frameworks:

- **Flower + TensorFlow**
- **TensorFlow Federated (TFF)**

The goal is to benchmark usability and behavior on distributed compute nodes (CloudLab).

---

## Flower + TensorFlow

### Environment
```bash
pip install flwr tensorflow
```

### Running
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

## TensorFlow Federated (TFF)
Unstarted

### Environment

### Running