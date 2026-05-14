import sys, os, time, datetime, subprocess, platform
import numpy as np
import cv2
import queue
import threading

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QGridLayout, QFrame, QSlider,
    QSizePolicy, QInputDialog, QScrollArea, QComboBox,
)
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QFont, QPolygonF, QPixmap, QImage,
)
from PyQt5.QtCore import (
    Qt, QTimer, QRect, QSize, QPointF, pyqtSignal, QThread,
)

# ─── Default RTSP URL (overridden by startup dialog) ─────────────────────────
RTSP_URL = "rtsp://your_camera_ip:554/stream"

# ─── Optional: sound player ──────────────────────────────────────────────────
try:
    from soundplayer import start_sound, stop_sound
except ImportError:
    def playsound():
        print("[SoundPlayer stub]  playsound()")

    def stopsound():
        print("[SoundPlayer stub]  stopsound()")

# ─── Optional: light controller ──────────────────────────────────────────────
try:
    from lightcontroller import (startIr, changeimagesettings,
                                  getCurrentImageSettings,
                                  movecamera, stopcamera)
except ImportError:
    def startIr(isOn=False):
        print(f"[LightController stub]  startlight(isOn={isOn})")

    def changeimagesettings(**kwargs):
        print(f"[LightController stub]  changeimagesettings({kwargs})")

    def getCurrentImageSettings():
        print("[LightController stub]  getCurrentImageSettings()")
        return {
            "BacklightCompensation": "OFF",
            "Brightness":            50.0,
            "ColorSaturation":       50.0,
            "Contrast":              50.0,
            "Focus":                 1.0,
            "Sharpness":             50.0,
        }

    def movecamera(direction: str):
        print(f"[LightController stub]  movecamera({direction!r})")

    def stopcamera():
        print("[LightController stub]  stopcamera()")

# ─── Wire detector function ───────────────────────────────────────────────────
try:
    from detecttrip import detect_wire as detectwire
    _HAS_DETECT = True
except ImportError:
    _HAS_DETECT = False
    def detectwire(frame):
        """Stub – returns None (detecttrip.py not found)."""
        return None

# ─── Palette ──────────────────────────────────────────────────────────────────
BG_DARK       = QColor("#0a0e1a")
BG_PANEL      = QColor("#0d1321")
BG_CARD       = QColor("#111827")
BORDER_COLOR  = QColor("#1e2d45")
ACCENT_BLUE   = QColor("#1a6bcc")
ACCENT_GREEN  = QColor("#22c55e")
ACCENT_YELLOW = QColor("#facc15")
ACCENT_RED    = QColor("#ef4444")
ACCENT_ORANGE = QColor("#f97316")
ACCENT_PURPLE = QColor("#a855f7")
TEXT_WHITE    = QColor("#f0f4ff")
TEXT_GRAY     = QColor("#6b7fa3")
TEXT_BLUE     = QColor("#60a5fa")


def lbl(text, color=TEXT_WHITE, size=9, bold=False):
    w = QLabel(text)
    w.setFont(QFont("Consolas", size, QFont.Bold if bold else QFont.Normal))
    p = w.palette(); p.setColor(w.foregroundRole(), color); w.setPalette(p)
    return w


def card_frame():
    f = QFrame()
    f.setStyleSheet(
        f"QFrame{{background:{BG_CARD.name()};"
        f"border:1px solid {BORDER_COLOR.name()};border-radius:6px;}}"
    )
    return f


# =============================================================================
#  Wi-Fi RSSI helper  (cross-platform, best-effort)
# =============================================================================
def _get_wifi_rssi() -> int:
    """Return signal quality 0-100.  Returns -1 on failure."""
    try:
        sys_platform = platform.system()
        if sys_platform == "Windows":
            out = subprocess.check_output(
                ["netsh", "wlan", "show", "interfaces"],
                timeout=2, stderr=subprocess.DEVNULL
            ).decode(errors="ignore")
            for line in out.splitlines():
                if "Signal" in line:
                    pct = int(line.split(":")[1].strip().replace("%", ""))
                    return pct
        elif sys_platform == "Linux":
            out = subprocess.check_output(
                ["iwconfig"], timeout=2, stderr=subprocess.DEVNULL
            ).decode(errors="ignore")
            for part in out.split():
                if "Quality=" in part:
                    q = part.split("=")[1].split("/")
                    return int(int(q[0]) / int(q[1]) * 100)
        elif sys_platform == "Darwin":
            # macOS
            out = subprocess.check_output(
                ["/System/Library/PrivateFrameworks/Apple80211.framework"
                 "/Versions/Current/Resources/airport", "-I"],
                timeout=2, stderr=subprocess.DEVNULL
            ).decode(errors="ignore")
            for line in out.splitlines():
                if "agrCtlRSSI" in line:
                    rssi = int(line.split(":")[1].strip())
                    # rssi typically -100..-30
                    return max(0, min(100, int((rssi + 100) * 1.43)))
    except Exception:
        pass
    return -1


# =============================================================================
#  CAMERA CAPTURE THREAD
#  – Captures frames from RTSP, auto-reconnects every 2 s.
#  – Applies tilt rotation to every frame before emitting.
#  – Writes raw frames to VideoWriter when recording is active.
#  – Generates a fast binary (Canny) frame for the detection mini-view.
#  – Puts a copy of raw frames into a queue consumed by DetectionThread.
# =============================================================================
class CameraThread(QThread):
    frame_ready    = pyqtSignal(QImage)   # tilt-corrected live frame → centre
    status_changed = pyqtSignal(str)      # "connecting" | "live" | "error"

    def __init__(self, url, frame_event: threading.Event, parent=None):
        super().__init__(parent)
        self.url             = url
        self._frame_event    = frame_event   # signalled whenever a new frame is ready
        self._latest_frame   = None          # most recent tilt-corrected frame (BGR ndarray)
        self._latest_lock    = threading.Lock()
        self._running        = False
        self._tilt_deg       = 0             # set by UI slider
        self._recording      = False
        self._rec_writer     = None
        self._rec_path       = ""
        self._do_start_rec   = False
        self._do_stop_rec    = False

    def get_latest_frame(self):
        """Thread-safe: return the most recent frame (or None)."""
        with self._latest_lock:
            return self._latest_frame

    # ── Thread-safe setters called from UI ────────────────────────────────
    def set_tilt(self, deg: int):
        self._tilt_deg = deg

    def request_start_record(self, path: str):
        self._rec_path     = path
        self._do_start_rec = True

    def request_stop_record(self):
        self._do_stop_rec = True

    # ── Helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _rotate(frame, deg):
        if deg == 0:
            return frame
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        return cv2.warpAffine(frame, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def _bgr_to_qimage(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

    # ── Main loop ──────────────────────────────────────────────────────────
    def run(self):
        self._running = True
        RETRY_DELAY   = 2.0            # seconds between reconnect attempts

        while self._running:
            self.status_changed.emit("connecting")
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                self.status_changed.emit("error")
                self._wait(RETRY_DELAY)
                cap.release()
                continue

            self.status_changed.emit("live")

            while self._running:
                ret, raw = cap.read()
                if not ret:
                    self.status_changed.emit("error")
                    break

                # Apply camera tilt
                frame = self._rotate(raw, self._tilt_deg)

                # ── Recording ────────────────────────────────────────────
                if self._do_start_rec and not self._recording:
                    self._do_start_rec = False
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self._rec_writer = cv2.VideoWriter(
                        self._rec_path, fourcc, 25.0, (w, h))
                    self._recording = True

                if self._do_stop_rec and self._recording:
                    self._do_stop_rec = False
                    self._recording = False
                    if self._rec_writer:
                        self._rec_writer.release()
                        self._rec_writer = None

                if self._recording and self._rec_writer:
                    self._rec_writer.write(frame)

                # ── Emit live frame ──────────────────────────────────────
                self.frame_ready.emit(self._bgr_to_qimage(frame))

                # ── Share frame with detection thread ────────────────────
                # Always overwrite with the freshest frame; the detection
                # thread will pick it up as soon as it finishes its current
                # detection — no queue accumulation at all.
                with self._latest_lock:
                    self._latest_frame = frame.copy()
                self._frame_event.set()   # wake detector immediately

                self.msleep(33)   # ~30 fps cap

            cap.release()
            if self._rec_writer:
                self._rec_writer.release()
                self._rec_writer = None
                self._recording  = False

            if self._running:
                self._wait(RETRY_DELAY)

    def _wait(self, secs):
        steps = int(secs / 0.1)
        for _ in range(steps):
            if not self._running:
                break
            self.msleep(100)

    def stop(self):
        self._running = False
        self._do_stop_rec = True
        self.wait(5000)
        if self._rec_writer:
            self._rec_writer.release()


# =============================================================================
#  DETECTION THREAD
#  – Waits on a threading.Event that CameraThread sets after every new frame.
#  – When the event fires, grabs the *latest* frame directly from CameraThread
#    (bypassing any queue) — so it never processes stale frames.
#  – Runs detectwire() which now returns points (not a marked image).
#  – Emits result_ready with (points, conf, angle, y_len, coverage) on a hit,
#    or None when nothing detected / below threshold.
#  – Never pauses; detection is continuous and self-regulating by speed.
# =============================================================================
class DetectionThread(QThread):
    result_ready = pyqtSignal(object)   # (points, conf, angle, y_len, coverage) | None
    binary_ready = pyqtSignal(QImage)   # binary frame produced inside detectwire

    def __init__(self, cam_thread, frame_event: threading.Event, parent=None):
        super().__init__(parent)
        self._cam         = cam_thread
        self._event       = frame_event
        self._running     = False
        self._sensitivity = 50

    def set_sensitivity(self, val: int):
        self._sensitivity = val

    # set_active kept for API compatibility — no longer pauses the thread,
    # but calling it with False is now a no-op (detection always runs).
    def set_active(self, active: bool):
        pass

    def _on_binary(self, binary_bgr_or_rgb):
        """Callback passed to detectwire(frame, callback)."""
        try:
            img = binary_bgr_or_rgb
            if img is None:
                return
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            qimg = QImage(img.data, w, h, ch * w, QImage.Format_RGB888).copy()
            self.binary_ready.emit(qimg)
        except Exception as exc:
            print(f"[DetectionThread] binary callback error: {exc}")

    def run(self):
        self._running = True
        while self._running:
            # Block until a new frame arrives (timeout so we can check _running)
            fired = self._event.wait(timeout=0.5)
            if not fired:
                continue

            # Clear the event BEFORE grabbing the frame so we don't miss the
            # next one that arrives while detection is running.
            self._event.clear()

            # Grab the freshest frame right now
            frame = self._cam.get_latest_frame()
            if frame is None:
                continue

            # Run detection — this may take 1-2 s; that is fine.
            # During that time CameraThread keeps overwriting _latest_frame
            # with fresh frames, and we'll pick up the newest one next cycle.
            try:
                result = detectwire(frame, self._on_binary)
            except TypeError:
                result = detectwire(frame)

            if result is None:
                self.result_ready.emit(None)
                continue

            points, confidence, angle, y_len, coverage = result
            threshold = self._sensitivity / 100.0
            if confidence >= threshold:
                self.result_ready.emit((points, confidence, angle, y_len, coverage))
            else:
                self.result_ready.emit(None)

    def stop(self):
        self._running = False
        self._event.set()   # unblock the wait so thread can exit
        self.wait(10000)


# =============================================================================
#  BINARY DETECTION VIEW  (right panel)
#  – Always square (width == height), driven by the panel width.
#  – Receives binary frames from DetectionThread.binary_ready at whatever
#    rate detectwire calls the callback (no extra timer needed).
#  – Center-crops the incoming image to fill the square exactly — no
#    letterboxing, no stretching.
# =============================================================================
class BinaryDetectionView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Square: expand horizontally to fill card, height matches width.
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap = None               # latest binary frame as QPixmap

    # Keep the widget square at all times
    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return w

    def sizeHint(self):
        return QSize(220, 220)

    def update_frame(self, qimg: QImage):
        """Receive a new binary QImage and repaint."""
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        side = min(self.width(), self.height())   # always square
        # Centre the square canvas inside the widget
        ox = (self.width()  - side) // 2
        oy = (self.height() - side) // 2
        dst = QRect(ox, oy, side, side)

        p.fillRect(self.rect(), QColor("#000000"))

        if self._pixmap:
            src_w = self._pixmap.width()
            src_h = self._pixmap.height()
            # Center-crop the source to a square, then draw into dst
            if src_w > src_h:
                crop_x = (src_w - src_h) // 2
                src = QRect(crop_x, 0, src_h, src_h)
            else:
                crop_y = (src_h - src_w) // 2
                src = QRect(0, crop_y, src_w, src_w)
            p.drawPixmap(dst, self._pixmap, src)
        else:
            p.setFont(QFont("Consolas", 7))
            p.setPen(TEXT_GRAY)
            p.drawText(dst, Qt.AlignCenter, "Waiting for\nbinary feed…")

        # Subtle scan-line texture
        scan = QPen(QColor(0, 0, 0, 35), 1)
        p.setPen(scan)
        for y in range(oy, oy + side, 3):
            p.drawLine(ox, y, ox + side, y)

        # Corner watermark
        p.setFont(QFont("Consolas", 6))
        p.setPen(QColor("#2a4060"))
        p.drawText(ox + 4, oy + 10, "BINARY / DETECT")
        p.end()


# =============================================================================
#  CENTRE CAMERA VIEW
#  Always shows the live RTSP feed.
#  When detection finds points they are drawn as red dots on every frame
#  until the next detection result clears them.
#  Points are stored in normalised [0,1] coordinates so they scale correctly
#  as the widget is resized.
# =============================================================================
class CameraFeed(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._live_pix       = None
        self._cam_status     = "connecting"
        self._blink          = True
        # Detection overlay — list of (norm_x, norm_y) in [0,1] coords,
        # or None when no wire is detected.
        self._det_points     = None   # type: list[tuple[float,float]] | None
        self._det_active     = False  # True while a wire is confirmed detected

        t = QTimer(self)
        t.timeout.connect(self._do_blink)
        t.start(600)

    def set_live_frame(self, qimg: QImage):
        self._live_pix   = QPixmap.fromImage(qimg)
        self._cam_status = "live"
        self.update()

    def set_cam_status(self, s: str):
        self._cam_status = s
        self.update()

    def set_detection_points(self, points, src_w: int, src_h: int):
        """
        Store detection points and convert to normalised coords.
        points: list/array of (x, y) pixel coords in the original frame space.
        src_w, src_h: dimensions of the frame detectwire ran on.
        """
        if points is None or len(points) == 0:
            self._det_points = None
            self._det_active = False
        else:
            self._det_points = [
                (float(px) / src_w, float(py) / src_h)
                for px, py in points
            ]
            self._det_active = True
        self.update()

    def clear_detection_points(self):
        """Remove overlay — called when consecutive clear detections."""
        self._det_points = None
        self._det_active = False
        self.update()

    # ── kept for backwards compat (manual-mode code path & old callers) ──
    def show_detected(self, qimg: QImage):
        pass   # no-op — we no longer freeze on a still frame

    def back_to_live(self):
        self.clear_detection_points()

    def _do_blink(self):
        self._blink = not self._blink
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()

        if self._live_pix:
            scaled = self._live_pix.scaled(w, h,
                                           Qt.KeepAspectRatioByExpanding,
                                           Qt.SmoothTransformation)
            ox = (scaled.width()  - w) // 2
            oy = (scaled.height() - h) // 2
            p.drawPixmap(0, 0, scaled, ox, oy, w, h)
        else:
            p.fillRect(0, 0, w, h, QColor("#0a0e1a"))
            msg = {
                "connecting": "Connecting to camera…",
                "error":      "⚠  Camera unavailable\nRetrying every 2 s…",
            }.get(self._cam_status, "Initialising…")
            p.setFont(QFont("Consolas", 11, QFont.Bold))
            p.setPen(TEXT_GRAY)
            p.drawText(QRect(0, 0, w, h), Qt.AlignCenter, msg)

        # ── Red detection point overlay ──────────────────────────────────
        if self._det_active and self._det_points:
            DOT_R    = 6
            HALO_R   = 11
            dot_col  = QColor("#ef4444")          # solid red dot
            halo_col = QColor(239, 68, 68, 70)    # translucent red halo
            for nx, ny in self._det_points:
                cx = int(nx * w)
                cy = int(ny * h)
                # Halo
                p.setBrush(halo_col); p.setPen(Qt.NoPen)
                p.drawEllipse(cx - HALO_R, cy - HALO_R, HALO_R * 2, HALO_R * 2)
                # Solid dot
                p.setBrush(dot_col)
                p.drawEllipse(cx - DOT_R, cy - DOT_R, DOT_R * 2, DOT_R * 2)
            # Blinking "WIRE DETECTED" banner while active
            if self._blink:
                p.fillRect(0, 0, w, 40, QColor(0, 0, 0, 180))
                p.setFont(QFont("Consolas", 11, QFont.Bold))
                p.setPen(ACCENT_RED)
                p.drawText(QRect(0, 0, w, 40), Qt.AlignCenter,
                           "◉  WIRE DETECTED")

        # ── CAMERA badge ────────────────────────────────────────────────
        dot = ACCENT_GREEN if self._cam_status == "live" else ACCENT_RED
        p.setBrush(QColor(0, 0, 0, 160))
        p.setPen(QPen(BORDER_COLOR, 1))
        p.drawRoundedRect(w - 115, 10, 105, 30, 4, 4)
        p.setFont(QFont("Consolas", 9, QFont.Bold)); p.setPen(TEXT_WHITE)
        p.drawText(QRect(w - 105, 10, 75, 30),
                   Qt.AlignVCenter | Qt.AlignLeft, "CAMERA")
        p.setBrush(dot); p.setPen(Qt.NoPen)
        p.drawEllipse(w - 24, 19, 10, 10)

        p.end()


# =============================================================================
#  SMALL WIDGETS
# =============================================================================
class BatteryWidget(QWidget):
    def __init__(self, percent=87, parent=None):
        super().__init__(parent)
        self.percent = percent
        self.setFixedSize(54, 22)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h   = self.width(), self.height()
        bw, bh = w - 6, h - 4
        p.setPen(QPen(ACCENT_GREEN, 1.5)); p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(0, 2, bw, bh, 2, 2)
        p.setBrush(ACCENT_GREEN); p.setPen(Qt.NoPen)
        p.drawRoundedRect(bw, h // 2 - 3, 5, 6, 1, 1)
        p.drawRoundedRect(2, 4, int((bw - 4) * self.percent / 100), bh - 4, 1, 1)
        p.end()


class WifiStrengthWidget(QWidget):
    """Animated Wi-Fi bar widget.  Call set_strength(0-100) to update."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._strength = 0
        self.setFixedSize(140, 28)
        # Poll Wi-Fi strength every 2 s in a background thread
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(2000)
        self._poll()

    def _poll(self):
        val = _get_wifi_rssi()
        if val >= 0:
            self._strength = val
        self.update()

    def set_strength(self, val: int):
        self._strength = max(0, min(100, val))
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        bars  = 10
        bw, gap, mh = 9, 4, self.height() - 2
        active = int(bars * self._strength / 100)
        for i in range(bars):
            bh = int(mh * (i + 1) / bars)
            if i < active:
                col = ACCENT_GREEN if self._strength >= 60 else (
                      ACCENT_YELLOW if self._strength >= 30 else ACCENT_RED)
            else:
                col = QColor("#1e3a2a")
            p.setBrush(col); p.setPen(Qt.NoPen)
            p.drawRoundedRect(i * (bw + gap), mh - bh + 1, bw, bh, 2, 2)
        p.end()

    def strength_label(self) -> str:
        if self._strength >= 70: return "STRONG"
        if self._strength >= 40: return "MODERATE"
        if self._strength >= 10: return "WEAK"
        return "NO SIGNAL"


class RulerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(48)

    def paintEvent(self, e):
        p   = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h      = self.width(), self.height()
        total_m   = 5.0
        margin    = 8
        draw_w    = w - 2 * margin
        p.fillRect(0, 0, w, h, QColor("#0b1120"))
        p.setFont(QFont("Consolas", 7))
        for m in range(6):
            x = margin + int(m / total_m * draw_w)
            p.setPen(QPen(TEXT_GRAY, 1))
            p.drawLine(x, 0, x, 14)
            p.drawText(x - 8, 26, 40, 14, Qt.AlignLeft, f"{m}m")
            for sub in range(1, 5):
                sx = margin + int((m + sub * 0.2) / total_m * draw_w)
                if sx < w - margin:
                    p.setPen(QPen(QColor("#2a3a55"), 1))
                    p.drawLine(sx, 0, sx, 7)
        p.end()


# =============================================================================
#  IMAGE SETTINGS PANEL
#  Calls changeimagesettings(**kwargs) on any change (debounced 600 ms).
#  Loads current values via getCurrentImageSettings() on construction.
# =============================================================================
class ImageSettingsPanel(QWidget):
    """
    Controls for:
      BacklightCompensation  – OFF / ON  (QComboBox)
      Brightness             – 0–100     (QSlider)
      ColorSaturation        – 0–100     (QSlider)
      Contrast               – 0–100     (QSlider)
      Focus                  – fixed 1.0 (MANUAL only; displayed read-only)
      Sharpness              – 0–100     (QSlider)

    Slider range stored ×10 internally for 0.1-step float precision.
    Every control change restarts a 600 ms debounce; on fire, all values
    are sent to changeimagesettings() in one call.
    """

    SLIDER_CSS = (
        "QSlider::groove:horizontal{background:#1a2540;height:3px;border-radius:2px;}"
        "QSlider::handle:horizontal{background:#60a5fa;width:12px;height:12px;"
        "margin:-5px 0;border-radius:6px;}"
        "QSlider::sub-page:horizontal{background:#1a6bcc;border-radius:2px;}"
        "QSlider::groove:horizontal:disabled{background:#111827;}"
        "QSlider::handle:horizontal:disabled{background:#1e2d45;}"
    )
    COMBO_CSS = (
        "QComboBox{background:#0d1321;color:#f0f4ff;border:1px solid #1e2d45;"
        "border-radius:4px;padding:2px 6px;font-family:Consolas;font-size:9pt;}"
        "QComboBox::drop-down{border:none;width:18px;}"
        "QComboBox::down-arrow{image:none;border:none;}"
        "QComboBox QAbstractItemView{background:#0d1321;color:#f0f4ff;"
        "selection-background-color:#1a6bcc;border:1px solid #1e2d45;}"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(600)
        self._debounce.timeout.connect(self._apply)

        main_ly = QVBoxLayout(self)
        main_ly.setContentsMargins(0, 0, 0, 0)
        main_ly.setSpacing(6)

        # ── Backlight Compensation ─────────────────────────────────────
        main_ly.addWidget(self._sec_lbl("BACKLIGHT COMPENSATION"))
        self._backlight = QComboBox()
        self._backlight.addItems(["OFF", "ON"])
        self._backlight.setStyleSheet(self.COMBO_CSS)
        self._backlight.currentIndexChanged.connect(self._schedule)
        main_ly.addWidget(self._backlight)

        # ── Numeric sliders ────────────────────────────────────────────
        # (label_text, dict_key, lo, hi, default, editable)
        slider_defs = [
            ("BRIGHTNESS",       "Brightness",      0,   100, 50,  True),
            ("COLOR SATURATION", "ColorSaturation", 0,   100, 50,  True),
            ("CONTRAST",         "Contrast",        0,   100, 50,  True),
            ("FOCUS  (manual)",  "Focus",           1,     1,  1,  False),  # fixed
            ("SHARPNESS",        "Sharpness",       0,   100, 50,  True),
        ]
        self._sliders: dict[str, tuple[QSlider, QLabel]] = {}
        for label_txt, key, lo, hi, default, editable in slider_defs:
            main_ly.addWidget(self._sec_lbl(label_txt))
            row = QHBoxLayout(); row.setSpacing(6)
            sl  = QSlider(Qt.Horizontal)
            sl.setRange(lo * 10, hi * 10)
            sl.setValue(int(default * 10))
            sl.setEnabled(editable)
            sl.setStyleSheet(self.SLIDER_CSS)
            val_lbl = lbl(f"{default:.0f}",
                          TEXT_BLUE if editable else TEXT_GRAY,
                          8, bold=True)
            val_lbl.setFixedWidth(30)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            def _make_cb(s=sl, vl=val_lbl):
                def cb(v):
                    vl.setText(f"{v / 10:.0f}")
                    self._schedule()
                return cb

            sl.valueChanged.connect(_make_cb())
            row.addWidget(sl); row.addWidget(val_lbl)
            main_ly.addLayout(row)
            self._sliders[key] = (sl, val_lbl)

        # ── Apply / Refresh buttons ────────────────────────────────────
        btn_row = QHBoxLayout(); btn_row.setSpacing(6)
        apply_btn = QPushButton("APPLY")
        apply_btn.setFont(QFont("Consolas", 8, QFont.Bold))
        apply_btn.setFixedHeight(28)
        apply_btn.setStyleSheet(
            f"QPushButton{{background:{ACCENT_BLUE.name()};color:white;"
            f"border-radius:4px;border:none;}}"
            f"QPushButton:hover{{background:#1a7ae0;}}"
        )
        apply_btn.clicked.connect(self._apply)

        refresh_btn = QPushButton("REFRESH")
        refresh_btn.setFont(QFont("Consolas", 8, QFont.Bold))
        refresh_btn.setFixedHeight(28)
        refresh_btn.setStyleSheet(
            f"QPushButton{{background:#1a2540;color:{TEXT_GRAY.name()};"
            f"border:1px solid {BORDER_COLOR.name()};border-radius:4px;}}"
            f"QPushButton:hover{{color:{TEXT_WHITE.name()};}}"
        )
        refresh_btn.clicked.connect(self.load_current)

        btn_row.addWidget(apply_btn); btn_row.addWidget(refresh_btn)
        main_ly.addLayout(btn_row)

        self.load_current()

    # ── Public ────────────────────────────────────────────────────────
    def load_current(self):
        try:
            settings = getCurrentImageSettings()
            if not isinstance(settings, dict):
                return
        except Exception as exc:
            print(f"[ImageSettings] getCurrentImageSettings() error: {exc}")
            return

        self._block(True)
        try:
            bl  = settings.get("BacklightCompensation", "OFF")
            idx = self._backlight.findText(str(bl).upper())
            if idx >= 0:
                self._backlight.setCurrentIndex(idx)
            for cam_key in ("Brightness", "ColorSaturation", "Contrast",
                            "Focus", "Sharpness"):
                if cam_key in settings and cam_key in self._sliders:
                    sl, vl = self._sliders[cam_key]
                    v = float(settings[cam_key])
                    sl.setValue(int(v * 10))
                    vl.setText(f"{v:.0f}")
        finally:
            self._block(False)

    # ── Internals ─────────────────────────────────────────────────────
    def _schedule(self):
        self._debounce.start()

    def _apply(self):
        kwargs = {
            "BacklightCompensation": self._backlight.currentText(),
            "Brightness":            self._sliders["Brightness"][0].value()      / 10.0,
            "ColorSaturation":       self._sliders["ColorSaturation"][0].value() / 10.0,
            "Contrast":              self._sliders["Contrast"][0].value()         / 10.0,
            "Focus":                 self._sliders["Focus"][0].value()            / 10.0,
            "Sharpness":             self._sliders["Sharpness"][0].value()        / 10.0,
        }
        try:
            changeimagesettings(kwargs)
        except Exception as exc:
            print(f"[ImageSettings] changeimagesettings() error: {exc}")

    def _block(self, yes: bool):
        self._backlight.blockSignals(yes)
        for sl, _ in self._sliders.values():
            sl.blockSignals(yes)

    @staticmethod
    def _sec_lbl(text: str) -> QLabel:
        w = QLabel(text)
        w.setFont(QFont("Consolas", 7))
        p = w.palette(); p.setColor(w.foregroundRole(), TEXT_GRAY); w.setPalette(p)
        return w


# =============================================================================
#  PTZ DIRECTION PAD
#  Press-and-hold → movecamera(dir); release → stopcamera().
# =============================================================================
class PtzPad(QWidget):
    """4-directional camera pan/tilt pad.  Works with press & release."""

    _BTN_CSS_IDLE = (
        f"QPushButton{{background:#0d1321;color:{TEXT_GRAY.name()};"
        f"border:1px solid {BORDER_COLOR.name()};border-radius:6px;"
        f"font-size:16px;font-weight:bold;}}"
        f"QPushButton:hover{{background:#1a2540;color:{TEXT_WHITE.name()};"
        f"border-color:{ACCENT_BLUE.name()};}}"
        f"QPushButton:pressed{{background:{ACCENT_BLUE.name()};color:white;}}"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)

        dirs = [
            ("▲", "up",    0, 1),
            ("◀", "left",  1, 0),
            ("▶", "right", 1, 2),
            ("▼", "down",  2, 1),
        ]
        for symbol, direction, row, col in dirs:
            btn = QPushButton(symbol)
            btn.setFixedSize(42, 36)
            btn.setStyleSheet(self._BTN_CSS_IDLE)
            btn.pressed.connect(self._make_press(direction))
            btn.released.connect(stopcamera)
            grid.addWidget(btn, row, col, Qt.AlignCenter)

        # Centre dot (cosmetic)
        centre = QLabel("●")
        centre.setAlignment(Qt.AlignCenter)
        centre.setStyleSheet(f"color:{BORDER_COLOR.name()};font-size:10px;")
        grid.addWidget(centre, 1, 1, Qt.AlignCenter)

    @staticmethod
    def _make_press(direction: str):
        def _handler():
            movecamera(direction)
        return _handler


# =============================================================================
#  LEFT PANEL
#  Top section  : status, mode, sensitivity, tilt, IR, record
#  Bottom section (scroll area): PTZ pad + image settings
# =============================================================================
class LeftPanel(QWidget):
    sensitivity_changed = pyqtSignal(int)
    tilt_changed        = pyqtSignal(int)
    record_start        = pyqtSignal(str)
    record_stop         = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(235)
        self.setStyleSheet(f"background-color:{BG_PANEL.name()};")

        # ── Outer layout: fixed controls on top, scroll area below ────
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Fixed top section ──────────────────────────────────────────
        top_widget = QWidget()
        top_widget.setStyleSheet(f"background-color:{BG_PANEL.name()};")
        ly = QVBoxLayout(top_widget)
        ly.setContentsMargins(10, 10, 10, 6)
        ly.setSpacing(7)

        # Status card
        self._status_card = card_frame()
        scl = QHBoxLayout(self._status_card)
        scl.setContentsMargins(8, 6, 8, 6)
        self._status_icon = QLabel("…")
        self._status_icon.setAlignment(Qt.AlignCenter)
        self._status_icon.setFixedSize(36, 36)
        scl.addWidget(self._status_icon)
        sv = QVBoxLayout(); sv.setSpacing(1)
        self._status_top = lbl("SYSTEM STATUS", TEXT_GRAY, 7)
        self._status_val = lbl("CONNECTING…", ACCENT_ORANGE, 9, bold=True)
        sv.addWidget(self._status_top); sv.addWidget(self._status_val)
        scl.addLayout(sv)
        ly.addWidget(self._status_card)
        self._apply_status("connecting")

        # Detection mode – Auto / Manual toggle
        self._detection_mode = "auto"   # "auto" | "manual"
        mode_card = card_frame()
        mode_ly   = QVBoxLayout(mode_card)
        mode_ly.setContentsMargins(8, 6, 8, 6); mode_ly.setSpacing(5)
        mode_ly.addWidget(lbl("DETECTION MODE", TEXT_GRAY, 7))
        mode_btn_row = QHBoxLayout(); mode_btn_row.setSpacing(5)
        self._auto_btn   = QPushButton("AUTO")
        self._manual_btn = QPushButton("MANUAL")
        for btn in (self._auto_btn, self._manual_btn):
            btn.setFont(QFont("Consolas", 8, QFont.Bold))
            btn.setFixedHeight(26)
        self._auto_btn.clicked.connect(lambda: self._set_detection_mode("auto"))
        self._manual_btn.clicked.connect(lambda: self._set_detection_mode("manual"))
        mode_btn_row.addWidget(self._auto_btn)
        mode_btn_row.addWidget(self._manual_btn)
        mode_ly.addLayout(mode_btn_row)
        ly.addWidget(mode_card)
        self._set_detection_mode("auto")   # initialise appearance

        # Sensitivity slider
        sens_card = card_frame(); sl = QVBoxLayout(sens_card)
        sl.setContentsMargins(8, 6, 8, 6)
        hr = QHBoxLayout()
        hr.addWidget(lbl("SENSITIVITY", TEXT_GRAY, 7)); hr.addStretch()
        self._sens_pct = lbl("50%", TEXT_BLUE, 9, bold=True)
        hr.addWidget(self._sens_pct); sl.addLayout(hr)
        self._sens_slider = QSlider(Qt.Horizontal)
        self._sens_slider.setRange(0, 100); self._sens_slider.setValue(50)
        self._sens_slider.setStyleSheet(self._slider_css())
        self._sens_slider.valueChanged.connect(self._on_sensitivity)
        sl.addWidget(self._sens_slider)
        ly.addWidget(sens_card)

        # Camera tilt slider
        tilt_card = card_frame(); tl = QVBoxLayout(tilt_card)
        tl.setContentsMargins(8, 6, 8, 6)
        tr = QHBoxLayout()
        tr.addWidget(lbl("CAMERA TILT", TEXT_GRAY, 7)); tr.addStretch()
        self._tilt_val = lbl("0°", TEXT_BLUE, 9, bold=True)
        tr.addWidget(self._tilt_val); tl.addLayout(tr)
        self._tilt_slider = QSlider(Qt.Horizontal)
        self._tilt_slider.setRange(-45, 45); self._tilt_slider.setValue(0)
        self._tilt_slider.setStyleSheet(self._slider_css())
        self._tilt_slider.valueChanged.connect(self._on_tilt)
        tl.addWidget(self._tilt_slider)
        ly.addWidget(tilt_card)

        # IR + Record card
        ctrl = card_frame(); ctl = QVBoxLayout(ctrl)
        ctl.setContentsMargins(8, 6, 8, 8); ctl.setSpacing(6)

        self._ir_on = True
        self._ir_btn = QPushButton("IR  ON")
        self._ir_btn.setFont(QFont("Consolas", 8, QFont.Bold))
        self._ir_btn.setFixedHeight(30)
        self._ir_btn.setStyleSheet(self._idle_btn_css(ACCENT_PURPLE))
        self._ir_btn.clicked.connect(self._toggle_ir)
        ctl.addWidget(self._ir_btn)

        self._recording   = False
        self._rec_blink   = True
        self._rec_start_t = 0.0
        self._rec_btn = QPushButton("⏺  RECORD")
        self._rec_btn.setFont(QFont("Consolas", 8, QFont.Bold))
        self._rec_btn.setFixedHeight(34)
        self._rec_btn.setStyleSheet(self._rec_idle_css())
        self._rec_btn.clicked.connect(self._toggle_record)
        ctl.addWidget(self._rec_btn)
        ly.addWidget(ctrl)

        outer.addWidget(top_widget)

        # ── Horizontal divider ────────────────────────────────────────
        div = QFrame(); div.setFrameShape(QFrame.HLine)
        div.setStyleSheet(f"color:{BORDER_COLOR.name()};")
        outer.addWidget(div)

        # ── Scrollable bottom section (PTZ + Image Settings) ─────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            f"QScrollArea{{background:{BG_PANEL.name()};border:none;}}"
            f"QScrollBar:vertical{{background:#0d1321;width:6px;border-radius:3px;}}"
            f"QScrollBar::handle:vertical{{background:#1e2d45;border-radius:3px;}}"
            f"QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0px;}}"
        )

        scroll_contents = QWidget()
        scroll_contents.setStyleSheet(f"background:{BG_PANEL.name()};")
        sc_ly = QVBoxLayout(scroll_contents)
        sc_ly.setContentsMargins(10, 8, 10, 10)
        sc_ly.setSpacing(10)

        # PTZ pad card
        ptz_card = card_frame()
        ptz_layout = QVBoxLayout(ptz_card)
        ptz_layout.setContentsMargins(8, 6, 8, 8); ptz_layout.setSpacing(6)
        ptz_hdr = QHBoxLayout()
        ptz_hdr.addWidget(lbl("CAMERA MOVEMENT", TEXT_WHITE, 8, bold=True))
        ptz_hdr.addStretch()
        ptz_hdr.addWidget(lbl("PTZ", TEXT_GRAY, 6))
        ptz_layout.addLayout(ptz_hdr)
        ptz_layout.addWidget(PtzPad(), alignment=Qt.AlignCenter)
        sc_ly.addWidget(ptz_card)

        # Image settings card
        img_card = card_frame()
        img_layout = QVBoxLayout(img_card)
        img_layout.setContentsMargins(8, 6, 8, 8); img_layout.setSpacing(6)
        img_hdr = QHBoxLayout()
        img_hdr.addWidget(lbl("IMAGE SETTINGS", TEXT_WHITE, 8, bold=True))
        img_hdr.addStretch()
        img_hdr.addWidget(lbl("CAM", TEXT_GRAY, 6))
        img_layout.addLayout(img_hdr)
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color:{BORDER_COLOR.name()};")
        img_layout.addWidget(sep)
        self._img_settings = ImageSettingsPanel()
        img_layout.addWidget(self._img_settings)
        sc_ly.addWidget(img_card)

        sc_ly.addStretch()
        scroll.setWidget(scroll_contents)
        outer.addWidget(scroll, stretch=1)

        # Timers
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._blink_record)
        self._blink_timer.start(500)
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed)

    # ── Status ────────────────────────────────────────────────────────
    def update_status(self, s: str):
        self._apply_status(s)

    def _apply_status(self, s: str):
        if s == "live":
            icon_txt, bg, border = "✓", "#0f2b1a", ACCENT_GREEN.name()
            val_txt, val_color   = "READY",        ACCENT_GREEN
        elif s == "error":
            icon_txt, bg, border = "✗", "#200808", ACCENT_RED.name()
            val_txt, val_color   = "ERROR",        ACCENT_RED
        else:
            icon_txt, bg, border = "…", "#1a1000", ACCENT_ORANGE.name()
            val_txt, val_color   = "CONNECTING…",  ACCENT_ORANGE

        self._status_icon.setText(icon_txt)
        self._status_icon.setStyleSheet(
            f"color:{border};background:{bg};border:2px solid {border};"
            f"border-radius:18px;padding:2px 6px;font-size:16px;font-weight:bold;"
        )
        self._status_val.setText(val_txt)
        p = self._status_val.palette()
        p.setColor(self._status_val.foregroundRole(), val_color)
        self._status_val.setPalette(p)

    # ── Sensitivity ───────────────────────────────────────────────────
    def _on_sensitivity(self, val: int):
        self._sens_pct.setText(f"{val}%")
        self.sensitivity_changed.emit(val)

    # ── Tilt ──────────────────────────────────────────────────────────
    def _on_tilt(self, val: int):
        self._tilt_val.setText(f"{val}°")
        self.tilt_changed.emit(val)

    # ── Detection Mode ────────────────────────────────────────────────
    def _set_detection_mode(self, mode: str):
        self._detection_mode = mode
        auto_active   = (mode == "auto")
        active_css = (
            f"QPushButton{{background:{ACCENT_BLUE.name()};color:white;"
            f"border-radius:4px;border:none;}}"
            f"QPushButton:hover{{background:#1a7ae0;}}"
        )
        idle_css = (
            f"QPushButton{{background:#0d1321;color:{TEXT_GRAY.name()};"
            f"border:1px solid {BORDER_COLOR.name()};border-radius:4px;}}"
            f"QPushButton:hover{{color:{TEXT_WHITE.name()};border-color:{ACCENT_BLUE.name()};}}"
        )
        self._auto_btn.setStyleSheet(active_css   if auto_active else idle_css)
        self._manual_btn.setStyleSheet(idle_css   if auto_active else active_css)

    @property
    def detection_mode(self) -> str:
        return self._detection_mode

    # ── IR ────────────────────────────────────────────────────────────
    def _toggle_ir(self):
        self._ir_on = not self._ir_on
        if self._ir_on:
            self._ir_btn.setText("IR  ON")
            self._ir_btn.setStyleSheet(self._active_btn_css(ACCENT_PURPLE))
        else:
            self._ir_btn.setText("IR  OFF")
            self._ir_btn.setStyleSheet(self._idle_btn_css(ACCENT_PURPLE))
        startIr(self._ir_on)

    # ── Record ────────────────────────────────────────────────────────
    def _toggle_record(self):
        if not self._recording:
            self._recording   = True
            self._rec_start_t = time.time()
            self._elapsed_timer.start(1000)
            path = f"recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            self.record_start.emit(path)
        else:
            self._recording = False
            self._elapsed_timer.stop()
            self._rec_btn.setStyleSheet(self._rec_idle_css())
            self._rec_btn.setText("⏺  RECORD")
            self.record_stop.emit()

    def _blink_record(self):
        if not self._recording:
            return
        self._rec_blink = not self._rec_blink
        if self._rec_blink:
            self._rec_btn.setStyleSheet(
                f"QPushButton{{background:{ACCENT_RED.name()};color:white;"
                f"border-radius:5px;font-weight:bold;}}")
        else:
            self._rec_btn.setStyleSheet(
                f"QPushButton{{background:#3a0000;color:{ACCENT_RED.name()};"
                f"border:1px solid {ACCENT_RED.name()};border-radius:5px;font-weight:bold;}}")

    def _update_elapsed(self):
        if not self._recording:
            return
        elapsed = int(time.time() - self._rec_start_t)
        m, s = divmod(elapsed, 60)
        self._rec_btn.setText(f"⏹  {m:02d}:{s:02d}  STOP")

    # ── CSS helpers ───────────────────────────────────────────────────
    @staticmethod
    def _slider_css():
        return (
            "QSlider::groove:horizontal{background:#1e2d45;height:4px;border-radius:2px;}"
            "QSlider::handle:horizontal{background:#60a5fa;width:14px;height:14px;"
            "margin:-5px 0;border-radius:7px;}"
            "QSlider::sub-page:horizontal{background:#1a6bcc;border-radius:2px;}"
        )

    @staticmethod
    def _active_btn_css(accent: QColor):
        return (
            f"QPushButton{{background:#0d1f0d;color:{accent.name()};"
            f"border:1px solid {accent.name()};border-radius:5px;font-weight:bold;}}"
        )

    @staticmethod
    def _idle_btn_css(accent: QColor):
        return (
            f"QPushButton{{background:#0d1321;color:{TEXT_GRAY.name()};"
            f"border:1px solid {BORDER_COLOR.name()};border-radius:5px;}}"
            f"QPushButton:hover{{color:{accent.name()};border-color:{accent.name()};}}"
        )

    @staticmethod
    def _rec_idle_css():
        return (
            f"QPushButton{{background:#1a0a0a;color:{ACCENT_RED.name()};"
            f"border:1px solid {ACCENT_RED.name()};border-radius:5px;}}"
            f"QPushButton:hover{{background:#2a0a0a;}}"
        )

    def _static_row(self, top_text, bot_text):
        c  = card_frame()
        rl = QHBoxLayout(c); rl.setContentsMargins(8, 4, 8, 4)
        lf = QVBoxLayout(); lf.setSpacing(2)
        lf.addWidget(lbl(top_text, TEXT_GRAY, 7))
        lf.addWidget(lbl(bot_text, TEXT_BLUE, 10, bold=True))
        rl.addLayout(lf); rl.addStretch()
        return c



# =============================================================================
#  RIGHT PANEL
# =============================================================================
class RightPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(245)
        self.setStyleSheet(f"background-color:{BG_PANEL.name()};")
        ly = QVBoxLayout(self)
        ly.setContentsMargins(10, 10, 10, 10); ly.setSpacing(9)

        # ── Detection View ─────────────────────────────────────────────
        dv  = card_frame(); dvl = QVBoxLayout(dv)
        dvl.setContentsMargins(6, 6, 6, 6); dvl.setSpacing(3)
        hdr = QHBoxLayout()
        hdr.addWidget(lbl("DETECTION VIEW", TEXT_WHITE, 8, bold=True))
        hdr.addStretch()
        hdr.addWidget(lbl("BINARY", TEXT_GRAY, 6))
        dvl.addLayout(hdr)
        self.bin_view = BinaryDetectionView()
        dvl.addWidget(self.bin_view)
        ly.addWidget(dv)

        # ── Signal Strength ────────────────────────────────────────────
        ss  = card_frame(); ssl = QVBoxLayout(ss)
        ssl.setContentsMargins(8, 6, 8, 6); ssl.setSpacing(4)
        ssl.addWidget(lbl("SIGNAL STRENGTH", TEXT_WHITE, 8, bold=True))
        self._sig_lbl  = lbl("SCANNING…", TEXT_GRAY, 11, bold=True)
        ssl.addWidget(self._sig_lbl)
        self._wifi_bar = WifiStrengthWidget()
        # Update the label whenever the bar polls
        self._wifi_bar._timer.timeout.connect(self._refresh_sig_label)
        ssl.addWidget(self._wifi_bar)
        ly.addWidget(ss)

        # ── Wire Info (hidden until detection) ─────────────────────────
        self._wire_card = card_frame()
        wil = QVBoxLayout(self._wire_card)
        wil.setContentsMargins(8, 6, 8, 6); wil.setSpacing(4)
        wil.addWidget(lbl("WIRE INFO", TEXT_WHITE, 8, bold=True))
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color:{BORDER_COLOR.name()};"); wil.addWidget(sep)

        grid = QGridLayout()
        grid.setHorizontalSpacing(4); grid.setVerticalSpacing(5)
        keys = ["TYPE", "CONFIDENCE", "ANGLE", "Y-LENGTH", "COVERAGE"]
        self._info_vals: dict[str, QLabel] = {}
        for i, k in enumerate(keys):
            grid.addWidget(lbl(k, TEXT_GRAY, 7), i, 0)
            v = lbl("—", TEXT_WHITE, 8, bold=True)
            v.setAlignment(Qt.AlignRight)
            self._info_vals[k] = v
            grid.addWidget(v, i, 1)
        wil.addLayout(grid)
        ly.addWidget(self._wire_card)
        self._wire_card.hide()   # hidden until detection

        ly.addStretch()

        # Initial signal label update
        self._refresh_sig_label()

    def _refresh_sig_label(self):
        txt = self._wifi_bar.strength_label()
        self._sig_lbl.setText(txt)
        col = (ACCENT_GREEN   if txt == "STRONG"   else
               ACCENT_YELLOW  if txt == "MODERATE" else
               ACCENT_RED     if txt == "WEAK"     else TEXT_GRAY)
        p = self._sig_lbl.palette()
        p.setColor(self._sig_lbl.foregroundRole(), col)
        self._sig_lbl.setPalette(p)

    def show_wire_info(self, confidence: float, angle: float,
                       y_len: float, coverage: float):
        self._info_vals["TYPE"].setText("Trip Wire")
        self._info_vals["CONFIDENCE"].setText(f"{confidence * 100:.1f}%")
        self._info_vals["ANGLE"].setText(f"{angle:.1f}°")
        self._info_vals["Y-LENGTH"].setText(f"{y_len:.1f} px")
        self._info_vals["COVERAGE"].setText(f"{coverage * 100:.1f}%")
        self._wire_card.show()

    def hide_wire_info(self):
        self._wire_card.hide()


# =============================================================================
#  TOP HEADER
# =============================================================================
class TopHeader(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(44)
        self.setStyleSheet(
            f"background-color:{BG_DARK.name()};"
            f"border-bottom:1px solid {BORDER_COLOR.name()};"
        )
        ly = QHBoxLayout(self); ly.setContentsMargins(16, 0, 16, 0)
        ly.addWidget(lbl("WIRE DETECTOR", TEXT_WHITE, 14, bold=True))
        ly.addWidget(lbl("  Model: WD-1000", TEXT_GRAY, 9))
        ly.addStretch()
        self._time_lbl = lbl("", TEXT_WHITE, 11, bold=True)
        self._time_lbl.setAlignment(Qt.AlignCenter)
        ly.addWidget(self._time_lbl)
        ly.addStretch()
        ly.addWidget(lbl("  ", TEXT_WHITE, 10))
        ly.addWidget(BatteryWidget(87))
        t = QTimer(self); t.timeout.connect(self._tick); t.start(1000)
        self._tick()

    def _tick(self):
        self._time_lbl.setText(
            datetime.datetime.now().strftime("%I:%M %p"))


# =============================================================================
#  BOTTOM TAB BAR
# =============================================================================
class TabBar(QWidget):
    live_view_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(52)
        self.setStyleSheet(f"background-color:{BG_PANEL.name()};")
        ly = QHBoxLayout(self); ly.setContentsMargins(10, 6, 10, 6); ly.setSpacing(6)

        tabs = [("📹  LIVE VIEW", True),
                ("📍  MAP VIEW",  False),
                ("⏱  HISTORY",   False),
                ("📄  REPORT",   False)]
        self._btns = []
        for text, active in tabs:
            btn = QPushButton(text)
            btn.setFont(QFont("Consolas", 9, QFont.Bold))
            btn.setFixedHeight(38)
            self._set_btn_active(btn, active)
            ly.addWidget(btn)
            self._btns.append(btn)

        self._btns[0].clicked.connect(self._on_live_view)

    def _set_btn_active(self, btn, active):
        if active:
            btn.setStyleSheet(
                f"QPushButton{{background:{ACCENT_BLUE.name()};color:white;"
                f"border-radius:6px;padding:0 18px;border:none;}}"
                f"QPushButton:hover{{background:#1a7ae0;}}"
            )
        else:
            btn.setStyleSheet(
                f"QPushButton{{background:transparent;color:{TEXT_GRAY.name()};"
                f"border-radius:6px;padding:0 18px;"
                f"border:1px solid {BORDER_COLOR.name()};}}"
                f"QPushButton:hover{{color:white;border-color:{ACCENT_BLUE.name()};}}"
            )

    def _on_live_view(self):
        # Re-activate LIVE VIEW appearance
        for i, btn in enumerate(self._btns):
            self._set_btn_active(btn, i == 0)
        self.live_view_clicked.emit()


# =============================================================================
#  MAIN WINDOW
# =============================================================================
class WireDetectorApp(QMainWindow):
    def __init__(self, rtsp_url):
        super().__init__()
        self.setWindowTitle("Wire Detector  –  WD-1000")
        self.setMinimumSize(1120, 700)
        self.setStyleSheet(f"background-color:{BG_DARK.name()};")

        # ── Shared frame-ready event (replaces the old queue) ─────────
        # CameraThread sets it on every new frame; DetectionThread waits on it.
        self._frame_event = threading.Event()

        # ── Camera thread ──────────────────────────────────────────────
        self._cam_thread = CameraThread(rtsp_url, self._frame_event, self)

        # ── Detection thread ───────────────────────────────────────────
        self._det_thread = DetectionThread(self._cam_thread, self._frame_event, self)

        # ── Sound / overlay state ──────────────────────────────────────
        self._sound_playing    = False   # True while playsound() is active
        self._clear_streak     = 0       # consecutive None results
        # How many consecutive "no wire" detections before stopping sound/overlay
        self._CLEAR_THRESHOLD  = 3

        # ── Build UI ───────────────────────────────────────────────────
        central = QWidget(); self.setCentralWidget(central)
        root    = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        root.addWidget(TopHeader())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0); body.setSpacing(0)

        # Left panel
        self._left = LeftPanel()
        body.addWidget(self._left)
        body.addWidget(self._vsep())

        # Centre
        centre_wrap = QVBoxLayout()
        centre_wrap.setContentsMargins(0, 0, 0, 0); centre_wrap.setSpacing(0)
        self._cam_feed = CameraFeed()
        centre_wrap.addWidget(self._cam_feed, stretch=1)
        centre_wrap.addWidget(RulerWidget())
        self._tab_bar = TabBar()
        centre_wrap.addWidget(self._tab_bar)
        cw = QWidget(); cw.setLayout(centre_wrap)
        body.addWidget(cw, stretch=1)

        body.addWidget(self._vsep())

        # Right panel
        self._right = RightPanel()
        body.addWidget(self._right)

        bw = QWidget(); bw.setLayout(body)
        root.addWidget(bw, stretch=1)

        # ── Wire-up signals ────────────────────────────────────────────
        # Camera → UI
        self._cam_thread.frame_ready.connect(self._cam_feed.set_live_frame)
        self._cam_thread.status_changed.connect(self._on_cam_status)

        # Detection → UI
        self._det_thread.result_ready.connect(self._on_detect_result)
        self._det_thread.binary_ready.connect(self._right.bin_view.update_frame)

        # Left panel controls
        self._left.sensitivity_changed.connect(self._det_thread.set_sensitivity)
        self._left.tilt_changed.connect(self._cam_thread.set_tilt)
        self._left.record_start.connect(self._cam_thread.request_start_record)
        self._left.record_stop.connect(self._cam_thread.request_stop_record)

        # Tab bar "LIVE VIEW" → resume
        self._tab_bar.live_view_clicked.connect(self._resume_live)

        # ── Start threads ──────────────────────────────────────────────
        self._cam_thread.start()
        self._det_thread.start()

    # ── Helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _vsep():
        f = QFrame(); f.setFrameShape(QFrame.VLine)
        f.setStyleSheet(f"color:{BORDER_COLOR.name()};"); return f

    def _on_cam_status(self, s: str):
        self._left.update_status(s)
        self._cam_feed.set_cam_status(s)

    def _on_detect_result(self, result):
        if self._left.detection_mode == "manual":
            # Manual mode: binary view updates (via binary_ready signal) but
            # main view and panels are untouched.
            return

        # ── AUTO mode ─────────────────────────────────────────────────
        if result is None:
            # Accumulate consecutive clear detections
            self._clear_streak += 1
            if self._clear_streak >= self._CLEAR_THRESHOLD:
                # Wire is gone — clear overlay, stop sound, hide info panel
                self._cam_feed.clear_detection_points()
                self._right.hide_wire_info()
                if self._sound_playing:
                    try:
                        stopsound()
                    except Exception as exc:
                        print(f"[SoundPlayer] stopsound() error: {exc}")
                    self._sound_playing = False
            return

        # Wire detected — reset the clear streak
        self._clear_streak = 0
        points, confidence, angle, y_len, coverage = result

        # Determine the source frame dimensions for normalisation.
        # get_latest_frame() gives us the frame detectwire just ran on.
        src_frame = self._cam_thread.get_latest_frame()
        if src_frame is not None:
            src_h, src_w = src_frame.shape[:2]
        else:
            # Fallback: use a reasonable default (normalise to 1×1)
            src_w, src_h = 1, 1

        # Draw red dots on the live view (persists across non-detected frames)
        self._cam_feed.set_detection_points(points, src_w, src_h)

        # Update wire info panel on every detection
        self._right.show_wire_info(confidence, angle, y_len, coverage)

        # Start sound only once per detection event
        if not self._sound_playing:
            try:
                playsound()
            except Exception as exc:
                print(f"[SoundPlayer] playsound() error: {exc}")
            self._sound_playing = True

    def _resume_live(self):
        """LIVE VIEW tab pressed — stop sound and clear any overlay."""
        if self._sound_playing:
            try:
                stopsound()
            except Exception as exc:
                print(f"[SoundPlayer] stopsound() error: {exc}")
            self._sound_playing = False
        self._clear_streak = 0
        self._cam_feed.clear_detection_points()
        self._right.hide_wire_info()

    def closeEvent(self, e):
        self._cam_thread.stop()
        self._det_thread.stop()
        super().closeEvent(e)


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    url="rtsp://admin:123456789!@192.168.1.100:554/StreamingChannels/101"

    rtsp_source = int(url.strip()) if url.strip().isdigit() else url.strip()
    win = WireDetectorApp(rtsp_source)
    win.showMaximized()
    sys.exit(app.exec_())