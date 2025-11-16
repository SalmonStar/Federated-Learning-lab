import sys
import requests
import tensorflow as tf
import numpy as np

# ---------- 1. Server address ----------
if len(sys.argv) > 1:
    server_addr = sys.argv[1]
else:
    server_addr = "127.0.0.1:8080"

SERVER = f"http://{server_addr}"
print(f"[Client] Connecting to server at {SERVER}")


# ---------- 2. Client local model (same as FLWR) ----------
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)


# ---------- helper: convert weights ----------
def weights_to_list(weights):
    return [w.tolist() for w in weights]

def list_to_weights(lst):
    return [np.array(w, dtype=np.float32) for w in lst]


# ---------- 3. Get global parameters from server ----------
def get_global_parameters():
    print("[Client] Getting global weights...")
    r = requests.get(SERVER + "/global_model")
    resp = r.json()
    if resp["status"] != "ok":
        print("[Client] No global model yet, using random init")
        return model.get_weights()
    return list_to_weights(resp["weights"])


# ---------- 4. Local training ----------
def local_train(global_weights):
    print("[Client] Setting weights and training locally...")
    model.set_weights(global_weights)
    model.fit(x_train, y_train, epochs=1, batch_size=32)
    return model.get_weights()


# ---------- 5. Upload updated parameters ----------
def upload_updated_parameters(new_weights):
    print("[Client] Uploading updated weights...")
    payload = {
        "weights": weights_to_list(new_weights),
        "num_examples": len(x_train)
    }
    r = requests.post(SERVER + "/upload_weights", json=payload)
    print("[Client] Upload response:", r.json())


# ---------- 6. Evaluate ----------
def evaluate(global_weights):
    model.set_weights(global_weights)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[Client] Eval loss={loss}, accuracy={acc}")


if __name__ == "__main__":
    # --- Real FL logic identical to your FLWR client ---
    global_params = get_global_parameters()
    updated_params = local_train(global_params)
    upload_updated_parameters(updated_params)
    evaluate(updated_params)

