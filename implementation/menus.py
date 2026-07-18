from abc import ABC, abstractmethod
import numpy as np
import time
import cv2

from widgets import Target
from globals import C_WHITE, MENU_DELAY

class Menu(ABC):

    @abstractmethod
    def update(self, cursor: (int, int)):
        pass

    @abstractmethod
    def draw(self, frame: np.ndarray):
        pass

    @abstractmethod
    def is_active(self) -> bool:
        pass

    @abstractmethod
    def reset(self):
        pass

class TargetMenu(Menu):
    """
    Menu that contains targets which can
    be select using the dwell time
    """

    def __init__(self, width, height, radius = 120):
        cx, cy = width // 2, height // 2
        self.targets =[Target(int(cx + radius * np.cos(i * np.pi / 2)),
            int(cy + radius * np.sin(i * np.pi / 2)),
            r=45, label=lbl)
            for i, lbl in enumerate(["A", "B", "C", "D"])]

    def update(self, cursor: (int, int)):
        # Draw targets based on position + dwell time
        for tgt in self.targets:
            tgt.update(cursor)

    def draw(self, frame: np.ndarray):
        # Draw targets
        for tgt in self.targets:
            tgt.draw(frame)

    def is_active(self) -> bool:
        return True # Has no exit condition

    def reset(self):
        for t in self.targets:
            t.dwell_t0  = None
            t.activated = False


class DragMenu(Menu):
    """
    Menu that contains multiple selection boxes,
    which can be select by swiping up with the cursor
    """

    def __init__(self, width, height, offset_y=100, size=60, n=4):
        self.x = 0
        self.current = -1
        self.delta_y = 0
        self.prev_y = None
        self.offset_y = height - offset_y
        self.width = width
        self.height = height
        self.hsize=size//2
        self.n = n
        self.start_ts = None

    def update(self, cursor: (int, int)):

        # Wait atleast 500ms until input (prevent accidental input)
        if self.start_ts is None:
            self.start_ts = time.time()
            return

        if time.time() - self.start_ts < MENU_DELAY:
            return

        # Calculate box offset
        cx, cy = cursor

        if self.prev_y is not None:
            self.delta_y = max(0, self.delta_y + (self.prev_y - cy))

        # Calculate current box selection
        index = np.floor(self.n * cx / self.width)
        if self.current != index:
            self.current = index
            self.delta_y = 0

        self.prev_y = cy

    def draw(self, frame: np.ndarray):
        # Draw boxes evenly spaces on x axis
        for i in range(self.n):
            cbx = ((i + 0.5) / self.n) * self.width

            # Highlight/Offset current selection
            bdy = self.delta_y * 0.25 if i == self.current else 0
            color = C_WHITE if i == self.current else (160,160,160)
            bsize = 8 if i == self.current else 2

            cv2.rectangle(frame,
                          (int(cbx - self.hsize), int(self.offset_y - self.hsize - bdy)),
                          (int(cbx + self.hsize), int(self.offset_y + self.hsize - bdy)),
                          color, bsize)

    def is_active(self) -> bool:
        # If box moved sufficiently exit
        return self.delta_y * 2.5 < self.height

    def reset(self):
        self.x = 0
        self.current = -1
        self.delta_y = 0
        self.start_ts = None

