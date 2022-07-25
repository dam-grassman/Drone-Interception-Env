from math import pi
from flask import Flask, jsonify
from acas_env import AcasEnv
import argparse
from stable_baselines3 import TD3
from acas_wrappers import AcasWrapper


def state_to_dict(state, reward=0, done=False):
    state_dict = {
        "attacker_x": state[0],
        "attacker_y": state[1],
        "attacker_speed": float(state[2]),
        "attacker_heading": state[3],
        "target_x": state[4],
        "target_y": state[5],
        "target_speed": float(state[6]),
        "target_heading": state[7],
        "acas_state": int(state[8]),
        "win_zone_x": state[9],
        "win_zone_y": state[10],
        "loose_zone_x": float(state[11]),
        "loose_zone_y": float(state[12]),
    }

    state_dict["reward"] = reward
    state_dict["done"] = int(done)
    return state_dict

# Flask API
app = Flask(__name__)

# Argparser : randomness, spee, model_path, table (surrogate or ACAS Table)
parser = argparse.ArgumentParser(description="Acas env API")
parser.add_argument(
    "--randomness", type=float, default=0, help="Initial positions randomness"
)
parser.add_argument(
    "--speed", type=int, default=400, help="acas speed"
)
parser.add_argument(
    "--model_path", type=str, default="MODELS/rl_model/full_vatt_1000.zip", help="model path"
)

parser.add_argument(
    "--table", type=str, default="surrogate", help="acas or surrogate"
)
args = vars(parser.parse_args())

# ACAS Environment
env = AcasEnv(acs=args["table"], vtarget=args["speed"], randomness=args["randomness"])
env = AcasWrapper(env)
model = TD3.load(args["model_path"])
print("Model loaded")
state = None

@app.route("/")
def super_endpoint():
    return "Acas ENV API"


@app.route("/reset")
def reset():
    global state
    state = env.reset()
    state_dict = state_to_dict(env.state)
    return jsonify(state_dict)


@app.route("/step")
def step():
    global state
    action, _s_tates = model.predict(state)
    state, reward, done, info = env.step(action)
    state_dict = state_to_dict(env.state, float(reward), done)
    state_dict = jsonify(state_dict)
    return state_dict

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002)
    pass
