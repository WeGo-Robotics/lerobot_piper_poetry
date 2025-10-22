import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from torch import tensor

X_AXIS = "timestamp"

def interpolation(x, y, desam_cnt = 4):
    # 이동평균으로 노이즈 제거 후 선형 보간
    series = pd.Series(y, index=x)
    smoothed_signal = series.rolling(window=3, center=True).mean().tolist()
    smoothed_signal[0] = y[0]
    smoothed_signal[-1] = y[-1]

    # 중간 포인트들로 스플라인 만들어서 보간
    f_interp_loss = interp1d(x, smoothed_signal, kind='linear')
    x_loss = np.linspace(x[0], x[-1], num=desam_cnt, endpoint=True)
    y_interpolated_loss = f_interp_loss(x_loss)

    f_interp_orig = interp1d(x_loss, y_interpolated_loss, kind='cubic')
    x_orig = np.linspace(x[0], x[-1], num=len(x), endpoint=True)
    y_interpolated_array = f_interp_orig(x_orig)

    x_interpolated_list = x_orig.tolist()
    y_interpolated_list = y_interpolated_array.tolist()
    return x_interpolated_list, y_interpolated_list

# Load the logged actions from the YAML file
with open("robot_actions_log.yaml", "r") as file:
    log_data = yaml.safe_load(file)

with open("robot_observations_log.yaml", "r") as file:
    obs_data = yaml.safe_load(file)

timed_actions = log_data["timed_actions"]
timed_actions.pop(-1)
timed_observations = obs_data["obs"]
timed_observations.pop(-1)

data = {
    "group": [],
    "timestamp": [],
    "timestep": [],
    "joint0": [],
    "joint1": [],
    "joint2": [],
    "joint3": [],
    "joint4": [],
    "joint5": [],
    "gripper": [],
}

for v in timed_observations:
    data["group"].append("observation")
    data["timestamp"].append(v["timestamp"])
    data["timestep"].append(v["timestep"])
    obs = v["observation"]
    data["joint0"].append(obs["joint1"])
    data["joint1"].append(obs["joint2"])
    data["joint2"].append(obs["joint3"])
    data["joint3"].append(obs["joint4"])
    data["joint4"].append(obs["joint5"])
    data["joint5"].append(obs["joint6"])
    data["gripper"].append(obs["gripper"])

iter_idx = 0
for i, chunk in enumerate(timed_actions):
    chunk_data = {
        "group": [],
        "timestamp": [],
        "timestep": [],
        "joint0": [],
        "joint1": [],
        "joint2": [],
        "joint3": [],
        "joint4": [],
        "joint5": [],
        "gripper": [],
    }
    for _, iter in enumerate(chunk["iter"]):
        chunk_data["group"].append(i)
        chunk_data["timestamp"].append(float(iter["timestamp"]))
        chunk_data["timestep"].append(int(iter["timestep"]))
        actions = eval(iter["action"])
        actions = actions.numpy()
        chunk_data["joint0"].append(actions[0])
        chunk_data["joint1"].append(actions[1])
        chunk_data["joint2"].append(actions[2])
        chunk_data["joint3"].append(actions[3])
        chunk_data["joint4"].append(actions[4])
        chunk_data["joint5"].append(actions[5])
        chunk_data["gripper"].append(actions[6])
    # Orioginal data
    data["group"].extend(chunk_data["group"])
    data["timestamp"].extend(chunk_data["timestamp"])
    data["timestep"].extend(chunk_data["timestep"])
    data["joint0"].extend(chunk_data["joint0"])
    data["joint1"].extend(chunk_data["joint1"])
    data["joint2"].extend(chunk_data["joint2"])
    data["joint3"].extend(chunk_data["joint3"])
    data["joint4"].extend(chunk_data["joint4"])
    data["joint5"].extend(chunk_data["joint5"])
    data["gripper"].extend(chunk_data["gripper"])
    # Cubic interpolation
    for val in chunk_data["group"]:
        data["group"].append("cubic_" + str(val))
    data["timestep"].extend(chunk_data["timestep"])

    inpd_x, inpd_y = interpolation(chunk_data[X_AXIS], chunk_data["joint0"])
    data["timestamp"].extend(inpd_x)
    data["joint0"].extend(inpd_y)
    _, inpd_y = interpolation(chunk_data[X_AXIS], chunk_data["joint1"])
    data["joint1"].extend(inpd_y)
    _, inpd_y = interpolation(chunk_data[X_AXIS], chunk_data["joint2"])
    data["joint2"].extend(inpd_y)
    _, inpd_y = interpolation(chunk_data[X_AXIS], chunk_data["joint3"])
    data["joint3"].extend(inpd_y)
    _, inpd_y = interpolation(chunk_data[X_AXIS], chunk_data["joint4"])
    data["joint4"].extend(inpd_y)
    _, inpd_y = interpolation(chunk_data[X_AXIS], chunk_data["joint5"])
    data["joint5"].extend(inpd_y)
    _, inpd_y = interpolation(chunk_data[X_AXIS], chunk_data["gripper"])
    data["gripper"].extend(inpd_y)

    # Moving average smoothing
    for val in chunk_data["group"]:
        data["group"].append("filter_" + str(val))
    data["timestamp"].extend(chunk_data["timestamp"])
    data["timestep"].extend(chunk_data["timestep"])

    joint0_series = pd.Series(chunk_data["joint0"], index=chunk_data[X_AXIS])
    smoothed_signal = joint0_series.rolling(window=10, center=True).mean().tolist()
    data["joint0"].extend(smoothed_signal)
    joint1_series = pd.Series(chunk_data["joint1"], index=chunk_data[X_AXIS])
    smoothed_signal = joint1_series.rolling(window=5, center=True).mean().tolist()
    data["joint1"].extend(smoothed_signal)
    joint2_series = pd.Series(chunk_data["joint2"], index=chunk_data[X_AXIS])
    smoothed_signal = joint2_series.rolling(window=5, center=True).mean().tolist()
    data["joint2"].extend(smoothed_signal)
    joint3_series = pd.Series(chunk_data["joint3"], index=chunk_data[X_AXIS])
    smoothed_signal = joint3_series.rolling(window=5, center=True).mean().tolist()
    data["joint3"].extend(smoothed_signal)
    joint4_series = pd.Series(chunk_data["joint4"], index=chunk_data[X_AXIS])
    smoothed_signal = joint4_series.rolling(window=5, center=True).mean().tolist()
    data["joint4"].extend(smoothed_signal)
    joint5_series = pd.Series(chunk_data["joint5"], index=chunk_data[X_AXIS])
    smoothed_signal = joint5_series.rolling(window=5, center=True).mean().tolist()
    data["joint5"].extend(smoothed_signal)
    gripper_series = pd.Series(chunk_data["gripper"], index=chunk_data[X_AXIS])
    smoothed_signal = gripper_series.rolling(window=5, center=True).mean().tolist()
    data["gripper"].extend(smoothed_signal)
data_frame = pd.DataFrame(data)

# # Seaborn의 lineplot 함수를 사용
# plt.figure()
# sns.lineplot( x="timestamp", y="joint0", hue="group", data=data_frame, marker='o' )
# plt.title("Joint 0 Actions Over Time")

# plt.figure()
# sns.lineplot( x="timestamp", y="joint1", hue="group", data=data_frame, marker='o' )
# plt.title("Joint 1 Actions Over Time")

# plt.figure()
# sns.lineplot( x="timestamp", y="joint2", hue="group", data=data_frame, marker='o' )
# plt.title("Joint 2 Actions Over Time")

# plt.figure()
# sns.lineplot( x="timestamp", y="joint3", hue="group", data=data_frame, marker='o' )
# plt.title("Joint 3 Actions Over Time")

# plt.figure()
# sns.lineplot( x="timestamp", y="joint4", hue="group", data=data_frame, marker='o' )
# plt.title("Joint 4 Actions Over Time")

# plt.figure()
# sns.lineplot( x="timestamp", y="joint5", hue="group", data=data_frame, marker='o' )
# plt.title("Joint 5 Actions Over Time")

# plt.figure()
# sns.lineplot( x="timestamp", y="gripper", hue="group", data=data_frame, marker='o' )
# plt.title("Gripper Actions Over Time")
# plt.show()

fig = px.line(data_frame, x=X_AXIS, y="joint0", color="group", title="Joint 0 Actions Over Time")
fig.show()
fig = px.line(data_frame, x=X_AXIS, y="joint1", color="group", title="Joint 1 Actions Over Time")
fig.show()
fig = px.line(data_frame, x=X_AXIS, y="joint2", color="group", title="Joint 2 Actions Over Time")
fig.show()
fig = px.line(data_frame, x=X_AXIS, y="joint3", color="group", title="Joint 3 Actions Over Time")
fig.show()
fig = px.line(data_frame, x=X_AXIS, y="joint4", color="group", title="Joint 4 Actions Over Time")
fig.show()
fig = px.line(data_frame, x=X_AXIS, y="joint5", color="group", title="Joint 5 Actions Over Time")
fig.show()
fig = px.line(data_frame, x=X_AXIS, y="gripper", color="group", title="Gripper Actions Over Time")
fig.show()

wait = input("Press Enter to continue...")
