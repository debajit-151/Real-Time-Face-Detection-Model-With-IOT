"""
servo_controller.py — Serial bridge between Python and the ESP32 servo/PIR controller.

This module is designed for graceful fallback: if the ESP32 is not connected,
the face recognition system continues to work in pure-software mode with no errors.
"""

import serial
import serial.tools.list_ports
import threading
import time


class ServoController:
    """
    Manages serial communication with the ESP32 for servo tracking and PIR events.

    Usage:
        ctrl = ServoController()
        ctrl.connect()               # auto-detects ESP32 or does nothing
        ctrl.send_track(angle=95)     # face tracking
        ctrl.send_scan()              # resume sweep
        ctrl.send_stop()              # idle
        ctrl.disconnect()
    """

    BAUD_RATE = 115200
    # Common ESP32 USB-Serial chip identifiers
    ESP32_IDENTIFIERS = ["CP210", "CH340", "CH910", "SLAB", "Silicon Labs", "USB-SERIAL", "usbserial"]

    def __init__(self):
        self.serial_conn = None
        self.connected = False
        self.running = False

        # State reported by ESP32
        self.motion_detected = False
        self.current_angle = 90
        self.esp_state = "IDLE"          # IDLE / SCANNING / TRACKING

        # Rate limiting: don't spam serial with TRACK commands
        self._last_track_time = 0
        self._track_cooldown = 0.05      # 50ms minimum between track commands

        # Reader thread
        self._reader_thread = None

        # Callbacks (optional)
        self.on_motion = None            # Called with (bool) when motion status changes
        self.on_state_change = None      # Called with (str) when ESP state changes

    # ─── Connection ─────────────────────────────────────────

    def connect(self, port=None):
        """
        Connect to the ESP32. If port is None, auto-detect.
        Returns True if connected, False otherwise (graceful fallback).
        """
        if port is None:
            port = self._auto_detect_port()

        if port is None:
            print("[SERVO] No ESP32 detected. Running in software-only mode.")
            return False

        try:
            self.serial_conn = serial.Serial(port, self.BAUD_RATE, timeout=1)
            time.sleep(2)  # Wait for ESP32 to reset after serial connection
            self.connected = True
            self.running = True

            # Start reader thread
            self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
            self._reader_thread.start()

            # Flush any boot messages
            if self.serial_conn.in_waiting:
                self.serial_conn.read(self.serial_conn.in_waiting)

            print(f"[SERVO] Connected to ESP32 on {port}")
            return True

        except serial.SerialException as e:
            print(f"[SERVO] Failed to connect to {port}: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Cleanly shut down the serial connection."""
        self.running = False
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2)

        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.send_stop()
                time.sleep(0.1)
                self.serial_conn.close()
            except Exception:
                pass

        self.connected = False
        print("[SERVO] Disconnected.")

    def _auto_detect_port(self):
        """Scan system serial ports for an ESP32 device."""
        ports = serial.tools.list_ports.comports()
        for p in ports:
            desc = f"{p.description} {p.manufacturer or ''}".upper()
            for identifier in self.ESP32_IDENTIFIERS:
                if identifier.upper() in desc:
                    print(f"[SERVO] Auto-detected ESP32 on {p.device} ({p.description})")
                    return p.device
        return None

    # ─── Serial Reader (background thread) ──────────────────

    def _read_loop(self):
        """Continuously reads lines from ESP32 and updates internal state."""
        while self.running and self.connected:
            try:
                if self.serial_conn and self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        self._parse_message(line)
                else:
                    time.sleep(0.01)  # Prevent busy-wait
            except serial.SerialException:
                print("[SERVO] Serial connection lost.")
                self.connected = False
                break
            except Exception:
                time.sleep(0.01)

    def _parse_message(self, line):
        """Parse a line from the ESP32."""
        if line.startswith("MOTION:"):
            val = line.split(":")[1]
            self.motion_detected = (val == "1")
            if self.on_motion:
                self.on_motion(self.motion_detected)

        elif line.startswith("ANGLE:"):
            try:
                self.current_angle = int(line.split(":")[1])
            except ValueError:
                pass

        elif line.startswith("STATE:"):
            self.esp_state = line.split(":")[1]
            if self.on_state_change:
                self.on_state_change(self.esp_state)

    # ─── Commands (Python → ESP32) ──────────────────────────

    def send_track(self, angle):
        """
        Send a TRACK command to move the servo to the specified angle.
        Rate-limited to prevent serial flooding.
        """
        if not self.connected:
            return

        now = time.time()
        if now - self._last_track_time < self._track_cooldown:
            return  # Too soon, skip this command

        angle = max(30, min(150, int(angle)))  # Clamp to servo range

        try:
            self.serial_conn.write(f"TRACK:{angle}\n".encode())
            self._last_track_time = now
        except serial.SerialException:
            self.connected = False

    def send_scan(self):
        """Tell ESP32 to resume autonomous scanning."""
        if not self.connected:
            return
        try:
            self.serial_conn.write(b"SCAN\n")
        except serial.SerialException:
            self.connected = False

    def send_stop(self):
        """Tell ESP32 to stop (idle)."""
        if not self.connected:
            return
        try:
            self.serial_conn.write(b"STOP\n")
        except serial.SerialException:
            self.connected = False

    # ─── Face Position → Servo Angle Mapping ────────────────

    @staticmethod
    def face_center_to_angle(face_x, face_w, frame_width, current_angle,
                              fov=60, dead_zone=30):
        """
        Convert a face's horizontal center position in the frame to a servo angle.

        Args:
            face_x: X coordinate of the face bounding box (left edge)
            face_w: Width of the face bounding box
            frame_width: Total width of the camera frame in pixels
            current_angle: The servo's current angle
            fov: Horizontal field of view of the camera in degrees (default 60°)
            dead_zone: Pixel dead zone in center of frame where no adjustment happens

        Returns:
            target_angle (int): The new desired servo angle
        """
        face_center_x = face_x + face_w / 2
        frame_center_x = frame_width / 2

        offset_px = face_center_x - frame_center_x

        # If face is within dead zone, don't move
        if abs(offset_px) < dead_zone:
            return current_angle

        # Convert pixel offset to angle offset
        # Positive offset_px = face is to the RIGHT of center
        # On a servo: increasing angle typically pans RIGHT
        degrees_per_pixel = fov / frame_width
        angle_offset = offset_px * degrees_per_pixel

        # Invert if your servo is mounted the other way
        # (uncomment the next line if tracking goes the wrong direction)
        # angle_offset = -angle_offset

        target = current_angle + angle_offset

        # Clamp
        target = max(30, min(150, int(target)))
        return target
