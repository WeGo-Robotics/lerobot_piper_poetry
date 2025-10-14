import time
import tkinter as tk
from dataclasses import dataclass

import cv2
import draccus
import numpy as np
import yaml
from PIL import Image, ImageTk
from rich import print
from rich.console import Console
from rich.layout import Layout
from rich.table import Table


@dataclass
class CameraConfig:
    path_or_index : str | int = 0
    auto_exposure : float = 1.0
    auto_whitebalance : float = 0.0
    auto_forcus : float = 0.0
    exposure : float = 30.0
    gain : float = 30.0
    focus : float = 0.0
    temperature : float = 5600.0
    zoom : float = 0.0
    mode : float = 0.0
    fps : float = 30.0
    hue : float = 0.0

@dataclass
class ProgramConfig:
    camera_setting : list[CameraConfig]
    usercon : bool = False

class CameraCon:
    cap: cv2.VideoCapture
    config : CameraConfig
    current : CameraConfig

    def __init__(self, path_or_index):
        self.cap = cv2.VideoCapture(path_or_index)
        self.config = CameraConfig()
        self.current = CameraConfig(path_or_index = path_or_index)

    def read(self):
        return self.cap.read()

    def get_property(self):
        self.current.exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        self.current.gain = self.cap.get(cv2.CAP_PROP_GAIN)
        self.current.focus = self.cap.get(cv2.CAP_PROP_FOCUS)
        self.current.zoom = self.cap.get(cv2.CAP_PROP_ZOOM)
        self.current.temperature = self.cap.get(cv2.CAP_PROP_TEMPERATURE)
        self.current.auto_exposure = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        self.current.auto_whitebalance = self.cap.get(cv2.CAP_PROP_AUTO_WB)
        self.current.auto_forcus = self.cap.get(cv2.CAP_PROP_AUTOFOCUS)
        self.current.mode = self.cap.get(cv2.CAP_PROP_MODE)
        self.current.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current.hue = self.cap.get(cv2.CAP_PROP_HUE)

    def _increase(self, target, step = 1):
        self.cap.set(target, self.cap.get(target) + step)
        return self.cap.get(target)

    def _decrease(self, target, step = 1):
        self.cap.set(target, self.cap.get(target) - step)
        return self.cap.get(target)

    def set_config(self, cfg: CameraConfig):
        self.config = cfg

    def set_camera(self, cfg: CameraConfig):
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, cfg.auto_exposure)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, cfg.auto_whitebalance)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, cfg.auto_forcus)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, cfg.exposure)
        self.cap.set(cv2.CAP_PROP_TEMPERATURE, cfg.temperature)
        self.cap.set(cv2.CAP_PROP_FOCUS, cfg.focus)
        self.cap.set(cv2.CAP_PROP_GAIN, cfg.gain)
        self.cap.set(cv2.CAP_PROP_MODE, cfg.mode)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        self.cap.set(cv2.CAP_PROP_HUE, cfg.hue)

    def display_config(self):
        table = Table(title=f"Camera Config for '{self.config.path_or_index}'")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Time", time.asctime())
        table.add_row("Auto Exposure", str(self.current.auto_exposure))
        table.add_row("Auto Whitebalance", str(self.current.auto_whitebalance))
        table.add_row("Auto Focus", str(self.current.auto_forcus))
        table.add_row("Exposure", str(self.current.exposure))
        table.add_row("Gain", str(self.current.gain))
        table.add_row("Focus", str(self.current.focus))
        table.add_row("Temperature", str(self.current.temperature))
        table.add_row("Mode", str(self.current.mode))
        table.add_row("FPS", str(self.current.fps))
        table.add_row("Hue", str(self.current.hue))
        return table

def proc_usercon(cams : list[CameraCon]):
    root = tk.Tk()
    # Create a frame
    app = tk.Frame(root, bg="white")
    app.grid()
    # Create a label in the frame
    lmain = tk.Label(app)
    lmain.grid(row=0, column=0)
    lmain.after(1, lambda: lmain.focus_force())

    lsub = tk.Label(app)
    lsub.grid(row=0, column=1)

    console = Console()
    console.clear()

    layout = Layout()
    layout.split_row(
        Layout(name="left"),
        Layout(name="right")
    )

    cam_idx = 0

    def video_stream():
        cam = cams[cam_idx]
        _, frame = cam.read()
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)

        def dispaly_edgepic():
            edge_img = cv2.Canny(frame, 100, 200)
            edge_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2RGBA)
            edge_img = Image.fromarray(edge_img)
            edge_imgtk = ImageTk.PhotoImage(image=edge_img)
            lsub.imgtk = edge_imgtk
            lsub.configure(image=edge_imgtk)

        dispaly_edgepic()

        def display_imageinfo():
            nonlocal cam, frame
            table = Table(title="Image Info")
            table.add_column("Name", style="cyan")
            table.add_column("Value", style="magenta")
            table.add_row("Width", str(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            table.add_row("Height", str(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            table.add_row("Channels", str(frame.shape[2]))
            hist_r = cv2.calcHist([frame], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([frame], [2], None, [256], [0, 256])
            table.add_row("Histogram (R) Max", str(hist_r.max()))
            table.add_row("Histogram (G) Max", str(hist_g.max()))
            table.add_row("Histogram (B) Max", str(hist_b.max()))
            brightness = np.mean(frame)
            table.add_row("Brightness (Mean)", f"{brightness:.2f}")
            # Simple exposure estimation based on histogram spread
            # This is a heuristic and might not perfectly match camera's exposure setting
            exposure_estimate = (hist_r.std() + hist_g.std() + hist_b.std()) / 3
            table.add_row("Exposure Estimate (Std Dev)", f"{exposure_estimate:.2f}")

            return table

        cam.get_property()
        layout["left"].update(cam.display_config())
        layout["right"].update(display_imageinfo())
        console.print(layout)

        lmain.after(1, video_stream)

    def on_key_press(event):
        nonlocal cam_idx
        cam = cams[cam_idx]
        key = event.keysym.lower()
        match key:
            case 'q':
                cam._increase(cv2.CAP_PROP_EXPOSURE)
            case 'a':
                cam._decrease(cv2.CAP_PROP_EXPOSURE)
            case 'w':
                cam._increase(cv2.CAP_PROP_GAIN)
            case 's':
                cam._decrease(cv2.CAP_PROP_GAIN)
            case 'e':
                cam._increase(cv2.CAP_PROP_FOCUS)
            case 'd':
                cam._decrease(cv2.CAP_PROP_FOCUS)
            case 'r':
                cam._increase(cv2.CAP_PROP_ZOOM)
            case 'f':
                cam._decrease(cv2.CAP_PROP_ZOOM)
            case 't':
                cam._increase(cv2.CAP_PROP_TEMPERATURE, 100)
            case 'g':
                cam._decrease(cv2.CAP_PROP_TEMPERATURE, 100)
            case 'y':
                cam._increase(cv2.CAP_PROP_HUE)
            case 'h':
                cam._decrease(cv2.CAP_PROP_HUE)
            case 'x':
                root.destroy()
            case '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9':
                idx = int(key) - 1
                if idx < len(cams):
                    cam_idx = idx
            case 'z':
                save_camera_config(cams, 'camera_prop.yaml')

    root.bind("<Key>", on_key_press)
    video_stream()
    root.mainloop()

def save_camera_config(cams : list[CameraCon], path: str):
    cfg_list = []
    for cam in cams:
        cam.get_property()
        cfg_list.append(cam.current)

    with open(path, 'w') as f:
        yaml.dump({'camera_setting': [vars(c) for c in cfg_list]}, f)

@draccus.wrap()
def main(cfg: ProgramConfig):
    cam_list = []
    for v in cfg.camera_setting:
        cam = CameraCon(v.path_or_index)
        cam.set_config(v)
        cam.set_camera(v)
        cam.get_property()
        cam.display_config()
        cam_list.append(cam)

    if cfg.usercon:
        proc_usercon(cam_list)

if __name__ == "__main__":
    main()
