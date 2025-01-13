import copy
import sys

import numpy as np
import pygame

from enum import IntEnum


class DrawProperties:
    def __init__(self, scale: float = 1.0):
        self.scale = scale

        # Define properties assuming default window size of (500, 300).
        self.joy_center_radius = 15
        self.joy_center_color = (200, 150, 90)
        self.joy_outer_radius = 100
        self.joy_outer_color = (255, 255, 255)
        self.joy_outer_thickness = 2

        self.bg_color = (0, 0, 0)

        # Rescale every property.
        self._scale()

    def _scale(self) -> None:
        """Rescale all defined properties."""
        for attr in dir(self):
            # Skips dunder/magic methods, callables, and scale itself.
            if (
                attr.startswith("__")
                or callable(getattr(self, attr))
                or attr == "scale"
            ):
                continue

            # Skip color properties.
            if type(self.__dict__[attr]) == tuple:
                continue

            self.__dict__[attr] = int(self.__dict__[attr] * self.scale)


class JoystickButton(IntEnum):
    BUTTON_A = 0
    BUTTON_B = 1
    BUTTON_X = 2
    BUTTON_Y = 3
    BUTTON_LB = 4
    BUTTON_RB = 5
    BUTTON_BACK = 6
    BUTTON_START = 7
    BUTTON_LOGITECH = 8
    BUTTON_LS = 9
    BUTTON_RS = 10


class JoystickManager:
    def __init__(
        self, display: bool = True, width: int = 500, height: int = 300
    ) -> None:
        self._display_mode = display
        print("Initializing Joystick Manager PyGame display window.")
        # Use pygame for display mode.
        self._width = width
        self._height = height
        self._props = DrawProperties(scale=1.0)
        self._init_pygame()
        self._create_screen()

        self.left_xy = np.array([0.0, 0.0], dtype=np.float32)
        self.right_xy = np.array([0.0, 0.0], dtype=np.float32)

        self.feasibility_value = 0.0
        self.feasibility_gt = 0.0

        if self._device == "joystick":
            self.button_states = [False] * self._joystick.get_numbuttons()
        else:
            self.button_states = [False] * 11

        self.gaits = [
            "trotting",
            "pronking",
            "bounding",
            "pacing",
        ]
        self.gait_idx = 0
        self.gait_name = self.gaits[self.gait_idx]

        # copy of last button states
        self.last_button_states = copy.deepcopy(self.button_states)

    def _init_pygame(self) -> None:
        pygame.joystick.init()
        pygame.font.init()
        self._device = None

        joysticks = [
            pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())
        ]
        if len(joysticks) == 0:
            print("No joysticks found, switching to Keyboard.")
            self._device = "keyboard"
        else:
            print(f"Found {len(joysticks)} joysticks.")
            self._device = "joystick"

            # Only use the first detected joystick.
            self._joystick = joysticks[0]

    def _create_screen(self) -> None:
        self._screen = pygame.display.set_mode((self._width, self._height))

    def _draw(self) -> None:
        self._screen.fill(self._props.bg_color)

        left_center = (int(self._width // 2 - self._width * 0.25), self._height // 2)
        right_center = (int(self._width // 2 + self._width * 0.25), self._height // 2)

        # Draw left stick.
        pygame.draw.circle(
            self._screen,
            self._props.joy_outer_color,
            left_center,
            self._props.joy_outer_radius,
            self._props.joy_outer_thickness,
        )
        screen_left_x = left_center[0] + int(
            self.left_xy[0] * self._props.joy_outer_radius
        )
        screen_left_y = left_center[1] + int(
            self.left_xy[1] * self._props.joy_outer_radius
        )
        pygame.draw.circle(
            self._screen,
            self._props.joy_center_color,
            (screen_left_x, screen_left_y),
            self._props.joy_center_radius,
            0,
        )

        # Draw right stick.
        pygame.draw.circle(
            self._screen,
            self._props.joy_outer_color,
            right_center,
            self._props.joy_outer_radius,
            self._props.joy_outer_thickness,
        )
        screen_right_x = right_center[0] + int(
            self.right_xy[0] * self._props.joy_outer_radius
        )
        screen_right_y = right_center[1] + int(
            self.right_xy[1] * self._props.joy_outer_radius
        )
        pygame.draw.circle(
            self._screen,
            self._props.joy_center_color,
            (screen_right_x, screen_right_y),
            self._props.joy_center_radius,
            0,
        )

        # Shoing current gait
        font = pygame.font.Font(None, 36)
        display_text = font.render(f"Gait: {self.gait_name}", True, (255, 255, 255))
        display_text_rect = display_text.get_rect(
            center=(self._width // 2, self._height - 25)
        )
        self._screen.blit(display_text, display_text_rect)

        # Add feasibility value text
        feasibility_text = font.render(
            f"Feasibility: {self.feasibility_value:.2f} | {self.feasibility_gt:.2f}",
            True,
            (255, 255, 255),
        )
        feasibility_text_rect = feasibility_text.get_rect(center=(self._width // 2, 25))
        self._screen.blit(feasibility_text, feasibility_text_rect)

    def _update_pygame(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        _jb = JoystickButton

        if self._device == "joystick":
            self.left_xy[0] = self._joystick.get_axis(0)
            self.left_xy[1] = self._joystick.get_axis(1)
            self.right_xy[0] = self._joystick.get_axis(3)
            self.right_xy[1] = self._joystick.get_axis(4)
            self.button_states = [
                self._joystick.get_button(x)
                for x in range(self._joystick.get_numbuttons())
            ]
        else:
            keys = pygame.key.get_pressed()
            self.left_xy[0] = keys[pygame.K_d] - keys[pygame.K_a]
            self.left_xy[1] = keys[pygame.K_s] - keys[pygame.K_w]
            self.right_xy[0] = keys[pygame.K_e] - keys[pygame.K_q]

            self.button_states[5] = keys[pygame.K_SPACE]

        # detect only when 0 change to 1
        if (
            self.button_states[_jb.BUTTON_RB]
            and self.last_button_states[_jb.BUTTON_RB]
            != self.button_states[_jb.BUTTON_RB]
        ):
            self.gait_idx = (self.gait_idx + 1) % len(self.gaits)
            self.gait_name = self.gaits[self.gait_idx]

        self.last_button_states = copy.deepcopy(self.button_states)

        self._draw()

        pygame.display.flip()

    def update(self) -> None:
        self._update_pygame()


if __name__ == "__main__":
    import time

    mngr = JoystickManager(display=True)

    while True:
        t_start = time.perf_counter()
        mngr.update()
        t_end = time.perf_counter()
        elapsed = (t_end - t_start) * 1000
        print(f"Update time: {elapsed:.3f} ms.")
        time.sleep(1 / 60.0)

        # print(mngr.left_xy, mngr.right_xy)
