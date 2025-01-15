import tkinter as tk


class TrackBarSliders:
    def __init__(self):
        self._tk_root = tk.Tk()
        self._tk_root.title("WTW Trackbars")
        self._range_multiplier = 1.0
        self._sliders = []
        self._labels = []

        self._trackbars = [
            ["body_height", [-0.25, 0.15], 0.0],
            ["step_frequency", [2.0, 4.0], 3.0],
            ["gait_duration", [0.1, 1.0], 0.5],
            ["foot_swing", [0.03, 0.35], 0.08],
            ["body_pitch", [-0.4, 0.4], 0.0],
            ["body_roll", [-0.4, 0.4], 0.0],
            ["stance_width", [0.10, 0.45], 0.25],
            ["stance_length", [0.35, 0.45], 0.0],
        ]
        self._len_trackbars = len(self._trackbars)
        self.trackbar_values = {}

        for i in range(self._len_trackbars):
            tb_label, [tb_min_range, tb_max_range], tb_default_val = self._trackbars[i]
            label = tk.Label(self._tk_root, text=f"{tb_label}")
            label.pack()
            self._labels.append(label)

            slider = tk.Scale(
                self._tk_root,
                from_=tb_min_range,
                to=tb_max_range,
                length=150,
                resolution=0.01,
                orient=tk.HORIZONTAL,
                command=self.update_label,
            )
            slider.set(tb_default_val)
            slider.pack()
            self._sliders.append(slider)

            self.trackbar_values[tb_label] = tb_default_val

    # Function to update the label with the current slider value
    def update_label(self, *args):
        for i, slider in enumerate(self._sliders):
            raw_value = slider.get()
            tb_label = self._trackbars[i][0]
            self.trackbar_values[tb_label] = raw_value * self._range_multiplier

    def get_vals(self):
        return self.trackbar_values
