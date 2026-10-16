import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
import time
import math
from collections import deque


# =========================================================
# USER SETTINGS
# =========================================================

# Sway-box size in pixels
BOX_WIDTH_PIXELS = 200
BOX_HEIGHT_PIXELS = 200

# Calibration duration
CALIBRATION_DURATION_SECONDS = 10

# Activity thresholds in pixels/second
# These may need adjustment depending on camera distance.
STANDING_MAX_SPEED_PIXELS = 40
RUNNING_MIN_SPEED_PIXELS = 250

# Sidebar width
SIDEBAR_WIDTH = 360


# =========================================================
# COLORS
# =========================================================

PURPLE = (255, 0, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
GRAY = (80, 80, 80)
DARK_GRAY = (35, 35, 35)


# =========================================================
# MEDIAPIPE MODEL
# =========================================================

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)

script_folder = os.path.dirname(
    os.path.abspath(__file__)
)

model_file = os.path.join(
    script_folder,
    "pose_landmarker_full.task"
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def calculate_angle(point_a, point_b, point_c):
    """
    Calculates the angle at point_b.

    For a knee angle:
    point_a = hip
    point_b = knee
    point_c = ankle
    """

    vector_ba = np.array([
        point_a[0] - point_b[0],
        point_a[1] - point_b[1]
    ])

    vector_bc = np.array([
        point_c[0] - point_b[0],
        point_c[1] - point_b[1]
    ])

    magnitude_ba = np.linalg.norm(vector_ba)
    magnitude_bc = np.linalg.norm(vector_bc)

    if magnitude_ba == 0 or magnitude_bc == 0:
        return None

    cosine_angle = np.dot(vector_ba, vector_bc) / (
        magnitude_ba * magnitude_bc
    )

    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    return math.degrees(
        math.acos(cosine_angle)
    )


def pixel_distance(point_a, point_b):
    return math.dist(point_a, point_b)


def classify_activity(speed_pixels_per_second):
    """
    Approximate activity classifier.
    """

    if speed_pixels_per_second < STANDING_MAX_SPEED_PIXELS:
        return "STANDING"

    elif speed_pixels_per_second < RUNNING_MIN_SPEED_PIXELS:
        return "WALKING"

    else:
        return "RUNNING"


def draw_text(
    frame,
    text,
    location,
    color=WHITE,
    scale=0.6
):
    cv2.putText(
        frame,
        text,
        location,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2
    )


def sidebar_text(
    panel,
    text,
    y_position,
    color=WHITE,
    scale=0.58
):
    cv2.putText(
        panel,
        text,
        (20, y_position),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2
    )


# =========================================================
# DOWNLOAD MEDIAPIPE MODEL
# =========================================================

if not os.path.exists(model_file):

    print("Downloading pose model...")

    urllib.request.urlretrieve(
        MODEL_URL,
        model_file
    )

    print("Download complete!")


# =========================================================
# SET UP MEDIAPIPE
# =========================================================

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=model_file
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================================================
# BODY SEGMENTS
# =========================================================

connections = [

    (11, 12),        # shoulders

    (11, 13),        # left upper arm
    (13, 15),        # left forearm

    (12, 14),        # right upper arm
    (14, 16),        # right forearm

    (11, 23),        # left torso
    (12, 24),        # right torso

    (23, 24),        # hips

    (23, 25),        # left thigh
    (25, 27),        # left shank

    (24, 26),        # right thigh
    (26, 28),        # right shank

    (27, 29),        # left heel
    (29, 31),        # left foot

    (28, 30),        # right heel
    (30, 32)         # right foot
]


# =========================================================
# START CAMERA
# =========================================================

camera = cv2.VideoCapture(0)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

if not camera.isOpened():

    print("Could not open the webcam.")
    exit()


# =========================================================
# DISPLAY WINDOW
# =========================================================

WINDOW_NAME = "Pose and Motion Tracker"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

# Start in full-screen mode
cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

fullscreen = True


# =========================================================
# TRACKING VARIABLES
# =========================================================

# Calibration
calibration_start_time = None
calibration_samples = []
calibration_complete = False
sway_box_center = None

# Waist and ankle movement
previous_motion_points = {}
previous_motion_time = None

# Motion-speed smoothing
speed_history = deque(maxlen=12)
smoothed_speed_pixels = 0.0

# Activity
activity_label = "WAITING"


# =========================================================
# RUN PROGRAM
# =========================================================

start_time = time.monotonic()


with PoseLandmarker.create_from_options(
    pose_options
) as pose_detector:

    while True:

        success, frame = camera.read()

        if not success:

            print("Could not read webcam frame.")
            break


        height, width, _ = frame.shape


        # =================================================
        # DEFAULT SIDEBAR VALUES
        # =================================================

        left_thigh_text = "--"
        right_thigh_text = "--"
        left_shank_text = "--"
        right_shank_text = "--"

        left_knee_angle_text = "--"
        right_knee_angle_text = "--"

        waist_displacement_text = "--"
        speed_text = (
            str(round(smoothed_speed_pixels, 1))
            + " px/s"
        )

        calibration_status = "Waiting for pose"
        sway_status = "Not calibrated"
        legs_status = "Not visible"


        # =================================================
        # POSE DETECTION
        # =================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp = int(
            (time.monotonic() - start_time) * 1000
        )

        pose_result = pose_detector.detect_for_video(
            mp_image,
            timestamp
        )


        if pose_result.pose_landmarks:

            landmarks = pose_result.pose_landmarks[0]

            points = []


            # Convert normalized coordinates into pixels
            for landmark in landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                visibility = landmark.visibility

                points.append(
                    (x, y, visibility)
                )


            # =================================================
            # DRAW BODY JOINTS WITHOUT HEAD
            # =================================================

            body_joint_indices = list(range(11, 33))


            for index in body_joint_indices:

                x, y, visibility = points[index]

                if visibility > 0.5:

                    cv2.circle(
                        frame,
                        (x, y),
                        6,
                        RED,
                        -1
                    )


            # Draw body segments
            for joint_1, joint_2 in connections:

                x1, y1, visibility_1 = points[joint_1]
                x2, y2, visibility_2 = points[joint_2]

                if (
                    visibility_1 > 0.5
                    and visibility_2 > 0.5
                ):

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        GREEN,
                        3
                    )


            # =================================================
            # FIND WAIST CENTER
            # =================================================

            left_hip_x, left_hip_y, left_hip_visibility = (
                points[23]
            )

            right_hip_x, right_hip_y, right_hip_visibility = (
                points[24]
            )


            waist_center = None


            if (
                left_hip_visibility > 0.5
                and right_hip_visibility > 0.5
            ):

                waist_x = int(
                    (left_hip_x + right_hip_x) / 2
                )

                waist_y = int(
                    (left_hip_y + right_hip_y) / 2
                )

                waist_center = (
                    waist_x,
                    waist_y
                )


                # Draw purple waist dot
                cv2.circle(
                    frame,
                    waist_center,
                    11,
                    PURPLE,
                    -1
                )

                draw_text(
                    frame,
                    "Waist",
                    (waist_x + 15, waist_y),
                    color=PURPLE
                )


                # =================================================
                # CALIBRATE SWAY BOX
                # =================================================

                if not calibration_complete:

                    if calibration_start_time is None:

                        calibration_start_time = (
                            time.monotonic()
                        )

                        calibration_samples = []


                    calibration_elapsed = (
                        time.monotonic()
                        - calibration_start_time
                    )

                    calibration_remaining = max(
                        0,
                        CALIBRATION_DURATION_SECONDS
                        - calibration_elapsed
                    )


                    calibration_samples.append(
                        waist_center
                    )


                    if (
                        calibration_elapsed
                        < CALIBRATION_DURATION_SECONDS
                    ):

                        calibration_status = (
                            "Stand still: "
                            + str(
                                round(
                                    calibration_remaining,
                                    1
                                )
                            )
                            + " s"
                        )

                        activity_label = "CALIBRATING"


                    else:

                        # Average waist location during calibration
                        average_x = int(
                            np.mean([
                                point[0]
                                for point in calibration_samples
                            ])
                        )

                        average_y = int(
                            np.mean([
                                point[1]
                                for point in calibration_samples
                            ])
                        )

                        sway_box_center = (
                            average_x,
                            average_y
                        )

                        calibration_complete = True
                        calibration_status = "Complete"


                # =================================================
                # TRACK WAIST, KNEES, AND ANKLES
                # =================================================

                current_motion_points = {
                    "waist": waist_center
                }


                if points[27][2] > 0.5:

                    current_motion_points["left_ankle"] = (
                        points[27][0],
                        points[27][1]
                    )


                if points[28][2] > 0.5:

                    current_motion_points["right_ankle"] = (
                        points[28][0],
                        points[28][1]
                    )


                current_time = time.monotonic()


                if previous_motion_time is not None:

                    elapsed_time = (
                        current_time
                        - previous_motion_time
                    )


                    if elapsed_time > 0:

                        current_speeds = []


                        for name, current_point in (
                            current_motion_points.items()
                        ):

                            if name in previous_motion_points:

                                distance_pixels = (
                                    pixel_distance(
                                        previous_motion_points[name],
                                        current_point
                                    )
                                )

                                speed = (
                                    distance_pixels
                                    / elapsed_time
                                )

                                current_speeds.append(
                                    speed
                                )


                        if len(current_speeds) > 0:

                            # Use the fastest visible body point.
                            # This helps running register better
                            # because the ankles move faster than
                            # the waist.
                            current_motion_speed = max(
                                current_speeds
                            )

                            speed_history.append(
                                current_motion_speed
                            )

                            smoothed_speed_pixels = (
                                0.75
                                * smoothed_speed_pixels
                                + 0.25
                                * np.mean(speed_history)
                            )


                previous_motion_points = (
                    current_motion_points
                )

                previous_motion_time = current_time


                # =================================================
                # ACTIVITY LABEL
                # =================================================

                if calibration_complete:

                    activity_label = classify_activity(
                        smoothed_speed_pixels
                    )


                speed_text = (
                    str(round(smoothed_speed_pixels, 1))
                    + " px/s"
                )


                # =================================================
                # DRAW SWAY BOX
                # =================================================

                if (
                    calibration_complete
                    and sway_box_center is not None
                ):

                    box_center_x = sway_box_center[0]
                    box_center_y = sway_box_center[1]


                    box_left = (
                        box_center_x
                        - int(BOX_WIDTH_PIXELS / 2)
                    )

                    box_right = (
                        box_center_x
                        + int(BOX_WIDTH_PIXELS / 2)
                    )

                    box_top = (
                        box_center_y
                        - int(BOX_HEIGHT_PIXELS / 2)
                    )

                    box_bottom = (
                        box_center_y
                        + int(BOX_HEIGHT_PIXELS / 2)
                    )


                    waist_inside_box = (
                        box_left <= waist_center[0]
                        <= box_right
                        and
                        box_top <= waist_center[1]
                        <= box_bottom
                    )


                    if waist_inside_box:

                        box_color = GREEN
                        sway_status = "LOW SWAY"

                    else:

                        box_color = RED
                        sway_status = "HIGH SWAY"


                    cv2.rectangle(
                        frame,
                        (box_left, box_top),
                        (box_right, box_bottom),
                        box_color,
                        3
                    )


                    # Draw box center
                    cv2.circle(
                        frame,
                        sway_box_center,
                        6,
                        CYAN,
                        -1
                    )


                    # Waist displacement from box center
                    waist_displacement_pixels = (
                        pixel_distance(
                            waist_center,
                            sway_box_center
                        )
                    )

                    waist_displacement_text = (
                        str(round(waist_displacement_pixels, 1))
                        + " px"
                    )


                # =================================================
                # CHECK LEG VISIBILITY
                # =================================================

                required_leg_joints = [
                    23, 24, 25, 26, 27, 28
                ]

                legs_are_visible = all(
                    points[index][2] > 0.5
                    for index in required_leg_joints
                )


                if legs_are_visible:

                    legs_status = "Visible"


                    # Joint coordinates
                    left_hip = points[23][:2]
                    right_hip = points[24][:2]

                    left_knee = points[25][:2]
                    right_knee = points[26][:2]

                    left_ankle = points[27][:2]
                    right_ankle = points[28][:2]


                    # Segment lengths in pixels
                    left_thigh_pixels = pixel_distance(
                        left_hip,
                        left_knee
                    )

                    right_thigh_pixels = pixel_distance(
                        right_hip,
                        right_knee
                    )

                    left_shank_pixels = pixel_distance(
                        left_knee,
                        left_ankle
                    )

                    right_shank_pixels = pixel_distance(
                        right_knee,
                        right_ankle
                    )


                    left_thigh_text = (
                        str(round(left_thigh_pixels, 1))
                        + " px"
                    )

                    right_thigh_text = (
                        str(round(right_thigh_pixels, 1))
                        + " px"
                    )

                    left_shank_text = (
                        str(round(left_shank_pixels, 1))
                        + " px"
                    )

                    right_shank_text = (
                        str(round(right_shank_pixels, 1))
                        + " px"
                    )


                    # Knee angles
                    left_knee_angle = calculate_angle(
                        left_hip,
                        left_knee,
                        left_ankle
                    )

                    right_knee_angle = calculate_angle(
                        right_hip,
                        right_knee,
                        right_ankle
                    )


                    if left_knee_angle is not None:

                        left_knee_angle_text = (
                            str(round(left_knee_angle, 1))
                            + " deg"
                        )


                    if right_knee_angle is not None:

                        right_knee_angle_text = (
                            str(round(right_knee_angle, 1))
                            + " deg"
                        )


        else:

            activity_label = "NO PERSON DETECTED"
            calibration_status = "No pose detected"


        # =================================================
        # CAMERA VIEW TEXT
        # =================================================

        draw_text(
            frame,
            "Q: quit | R: recalibrate | F: fullscreen",
            (20, 40),
            scale=0.65
        )

        draw_text(
            frame,
            "Red dots = joints | Green lines = limbs",
            (20, 75),
            scale=0.6
        )

        draw_text(
            frame,
            "Purple dot = waist | Box = sway region",
            (20, 105),
            scale=0.6
        )


        # =================================================
        # CREATE SIDEBAR
        # =================================================

        sidebar = np.zeros(
            (height, SIDEBAR_WIDTH, 3),
            dtype=np.uint8
        )

        sidebar[:] = DARK_GRAY


        sidebar_text(
            sidebar,
            "POSE MEASUREMENTS",
            35,
            color=CYAN,
            scale=0.75
        )

        cv2.line(
            sidebar,
            (20, 50),
            (SIDEBAR_WIDTH - 20, 50),
            GRAY,
            2
        )


        # Activity
        sidebar_text(
            sidebar,
            "ACTIVITY",
            85,
            color=YELLOW,
            scale=0.62
        )

        activity_color = (
            GREEN
            if activity_label == "STANDING"
            else YELLOW
            if activity_label == "WALKING"
            else RED
            if activity_label == "RUNNING"
            else WHITE
        )

        sidebar_text(
            sidebar,
            activity_label,
            115,
            color=activity_color,
            scale=0.78
        )

        sidebar_text(
            sidebar,
            "Speed: " + speed_text,
            145
        )


        # Calibration
        sidebar_text(
            sidebar,
            "CALIBRATION",
            190,
            color=YELLOW,
            scale=0.62
        )

        sidebar_text(
            sidebar,
            calibration_status,
            220
        )


        # Sway information
        sidebar_text(
            sidebar,
            "SWAY",
            265,
            color=YELLOW,
            scale=0.62
        )

        sway_color = (
            GREEN
            if sway_status == "LOW SWAY"
            else RED
            if sway_status == "HIGH SWAY"
            else YELLOW
        )

        sidebar_text(
            sidebar,
            sway_status,
            295,
            color=sway_color,
            scale=0.72
        )

        sidebar_text(
            sidebar,
            "From box center: "
            + waist_displacement_text,
            325
        )


        # Box dimensions
        sidebar_text(
            sidebar,
            "SWAY BOX SIZE",
            370,
            color=YELLOW,
            scale=0.62
        )

        sidebar_text(
            sidebar,
            "Width: "
            + str(BOX_WIDTH_PIXELS)
            + " px",
            400
        )

        sidebar_text(
            sidebar,
            "Height: "
            + str(BOX_HEIGHT_PIXELS)
            + " px",
            430
        )


        # Segment lengths
        sidebar_text(
            sidebar,
            "SEGMENT LENGTHS",
            475,
            color=YELLOW,
            scale=0.62
        )

        sidebar_text(
            sidebar,
            "Left thigh: " + left_thigh_text,
            505
        )

        sidebar_text(
            sidebar,
            "Right thigh: " + right_thigh_text,
            535
        )

        sidebar_text(
            sidebar,
            "Left shank: " + left_shank_text,
            565
        )

        sidebar_text(
            sidebar,
            "Right shank: " + right_shank_text,
            595
        )


        # Angles
        sidebar_text(
            sidebar,
            "KNEE ANGLES",
            640,
            color=YELLOW,
            scale=0.62
        )

        sidebar_text(
            sidebar,
            "Left: " + left_knee_angle_text,
            670
        )

        sidebar_text(
            sidebar,
            "Right: " + right_knee_angle_text,
            700
        )


        sidebar_text(
            sidebar,
            "Legs: " + legs_status,
            745,
            color=GREEN if legs_status == "Visible" else RED,
            scale=0.6
        )

        sidebar_text(
            sidebar,
            "Lengths/displacement are pixels",
            775,
            color=CYAN,
            scale=0.55
        )


        # Combine camera view and sidebar
        combined_view = np.hstack(
            (frame, sidebar)
        )


        cv2.imshow(
            WINDOW_NAME,
            combined_view
        )


        # =================================================
        # KEYBOARD CONTROLS
        # =================================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


        elif key == ord("r"):

            # Reset calibration
            calibration_start_time = None
            calibration_samples = []
            calibration_complete = False
            sway_box_center = None

            # Reset movement tracking
            previous_motion_points = {}
            previous_motion_time = None
            speed_history.clear()
            smoothed_speed_pixels = 0.0

            activity_label = "WAITING"

            print(
                "Calibration reset. "
                "Stand still for 10 seconds."
            )


        elif key == ord("f"):

            fullscreen = not fullscreen

            if fullscreen:

                cv2.setWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN
                )

            else:

                cv2.setWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL
                )

                cv2.resizeWindow(
                    WINDOW_NAME,
                    1600,
                    850
                )


# =================================================
# CLOSE PROGRAM
# =================================================

camera.release()
cv2.destroyAllWindows()
