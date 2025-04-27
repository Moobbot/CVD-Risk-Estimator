import sys

sys.path.append("./")

from tri_2d_net.model import Model
from config import MODEL_CONFIG

def init_model():
    # Initialize model
    print("Initializing model...")
    model_config = {
        "dout": True,
        "lr": 1e-4,
        "num_workers": 32,
        "batch_size": 16,
        "restore_iter": 0,
        "total_iter": 1000,
        "model_name": "NLST-Tri2DNet",
        "prt_path": MODEL_CONFIG["CHECKPOINT_PATH"],
        "accumulate_steps": 2,
        "train_source": None,
        "val_source": None,
        "test_source": None,
    }
    model_config["save_name"] = "_".join(
        [
            "{}".format(model_config["model_name"]),
            "{}".format(model_config["dout"]),
            "{}".format(model_config["lr"]),
            "{}".format(model_config["batch_size"]),
        ]
    )

    return Model(**model_config)
