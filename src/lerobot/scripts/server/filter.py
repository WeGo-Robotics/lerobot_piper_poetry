import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from torch import tensor

from lerobot.scripts.server.helpers import (
    Action,
    TimedAction,
)

# def fill_action_gaps(incoming_actions: list[TimedAction], current: TimedAction):
#     result_actions = incoming_actions
#     first_step = incoming_actions[0].get_timestep()
#     if first_step > 0:
#         origin_action = TimedAction(incoming_actions[0].get_timestamp() ,first_step - 1, current.get_action())
#         result_actions.insert(0, origin_action)
#     return result_actions

def action_filter(incoming_actions: list[TimedAction]):
    timestamp=[]
    joint1_pos=[]
    joint2_pos=[]
    joint3_pos=[]
    joint4_pos=[]
    joint5_pos=[]
    joint6_pos=[]
    gripper_pos=[]

    for val in incoming_actions:
        timestamp.append(val.get_timestep())
        action = val.get_action().tolist()
        joint1_pos.append(action[0])
        joint2_pos.append(action[1])
        joint3_pos.append(action[2])
        joint4_pos.append(action[3])
        joint5_pos.append(action[4])
        joint6_pos.append(action[5])
        gripper_pos.append(action[6])
    _, joint1_interp = _interpolation(timestamp, joint1_pos)
    _, joint2_interp = _interpolation(timestamp, joint2_pos)
    _, joint3_interp = _interpolation(timestamp, joint3_pos)
    _, joint4_interp = _interpolation(timestamp, joint4_pos)
    _, joint5_interp = _interpolation(timestamp, joint5_pos)
    _, joint6_interp = _interpolation(timestamp, joint6_pos)
    _, gripper_interp = _interpolation(timestamp, gripper_pos)

    result_actions = incoming_actions.copy()
    for val in incoming_actions:
        val.action = tensor([
            joint1_interp.pop(0),
            joint2_interp.pop(0),
            joint3_interp.pop(0),
            joint4_interp.pop(0),
            joint5_interp.pop(0),
            joint6_interp.pop(0),
            gripper_interp.pop(0),
        ])
    return result_actions

def _interpolation(x, y, desam_cnt = 4):
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
