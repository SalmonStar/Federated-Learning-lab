# server.py  -- real TFF aggregation server
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

app = Flask(__name__)

# ---------------------------
# 1) Keras model structure
# ---------------------------
def create_keras_model():
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    return model

keras_model = create_keras_model()
initial_weights = keras_model.get_weights()
weight_specs = [tff.TensorType(w.dtype, w.shape) for w in initial_weights]

# Federated type: list of weights from CLIENTS
federated_weights_type = tff.FederatedType(weight_specs, tff.CLIENTS)

# ---------------------------
# 2) TFF: True Federated Averaging (just averaging weights)
# ---------------------------
@tff.federated_computation(federated_weights_type)
def tff_fedavg(client_weights):
    return tff.federated_mean(client_weights)

# ---------------------------
# Server State
# ---------------------------
global_weights = initial_weights
pending_updates = []   # list of list-of-numpy-arrays

def to_numpy(weights_list):
    return [np.array(w, dtype=np.float32) for w in weights_list]

def to_list(weights_numpy):
    return [w.tolist() for w in weights_numpy]


# ---------------------------
# 3) API: client fetch global weights
# ---------------------------
@app.route("/global_model", methods=["GET"])
def get_global_model():
    return jsonify({"status": "ok", "weights": to_list(global_weights)})


# ---------------------------
# 4) API: client upload updated weights
# ---------------------------
@app.route("/upload_weights", methods=["POST"])
def upload_weights():
    data = request.get_json()
    weights = to_numpy(data["weights"])
    pending_updates.append(weights)
    print(f"[Server] Received update #{len(pending_updates)}")
    return jsonify({"status": "received"})


# ---------------------------
# 5) API: run TFF aggregation
# ---------------------------
@app.route("/aggregate", methods=["POST"])
def aggregate():
    global global_weights, pending_updates

    if len(pending_updates) == 0:
        return jsonify({"status": "error", "message": "no client updates"})

    print(f"[Server] Running TFF Federated Averaging on {len(pending_updates)} clients")

    # Convert Python list → TFF eager values
    tff_input = [
        [tf.convert_to_tensor(w) for w in client_weights]
        for client_weights in pending_updates
    ]

    # Run true federated averaging
    new_weights = tff_fedavg(tff_input)

    # new_weights is a list-of-tensors
    global_weights = [w.numpy() for w in new_weights]

    pending_updates = []

    print("[Server] TFF FedAvg complete.")
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("[Server] TFF aggregation server running on 0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080)
