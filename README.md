# Federated Learning Framework with TensorFlow
This repository compares two Federated Learning (FL) frameworks:

- **Flower + TensorFlow**
- **TensorFlow Federated (TFF)**

The goal is to benchmark framework usability and behavior on distributed compute nodes (CloudLab), and later integrate security features such as adversarial clients, DP, and TEE protection.

---

## Project Structure
```bash
Federated-Learning-lab/
├── flwr/ # Flower + TensorFlow implementation
└── TFF/ # TensorFlow Federated implementation
```

---

## Quick Start

| Framework | Description | Guide |
|----------|------------|-------|
Flower + TensorFlow | FL with standard TF clients | 👉 [`flwr/README.md`](./flwr/README.md) |
TensorFlow Federated (TFF) | FL simulation environment with TF | 👉 [`TFF/README.md`](./TFF/README.md) |

> Each subfolder contains its own installation and execution guide.

---

## Status

| Component | Status |
|---|---|
Flower baseline | ✅ working |
TFF setup | 🔧 in progress |
Adversarial client | 🔜 |
DP experiment | 🔜 |
TEE integration | 🔜 (future CloudLab phase) |

Stay tuned. 🚀

