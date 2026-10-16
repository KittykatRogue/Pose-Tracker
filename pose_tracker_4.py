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
# Increase these values to make the box larger.
BOX_WIDTH_PIXELS = 200
BOX_HEIGHT_PIXELS = 200

# Calibration duration
CALIBRATION_DURATION_SECONDS = 10

# Activity thresholds in pixels/second
# These may need adjustment depending on camera distance.
STANDING_MAX_SPEED_PIXELS = 20
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
BLACK = (0, 0, 0)
GRAY = (80, 80, 80)


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
    Classifies motion using waist speed.

    This is an approximate classifier.
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
# TRACKING VARIABLES
# =========================================================

# Ten-second calibration
calibration_start_time = None
calibration_samples = []
calibration_complete = False
sway_box_center = None

# Waist tracking
previous_waist_center = None
previous_waist_time = None

# Speed smoothing
speed_history = deque(maxlen=15)
smoothed_speed_pixels = 0.0

# Displacement
total_displacement_pixels = 0.0

# Movement trail
movement_trail = []

# Activity
activity_label = "WAITING"

# Time
start_time = time.monotonic()


# =========================================================
# RUN PROGRAM
# =========================================================

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
        # DEFAULT MEASUREMENT VALUES
        # =================================================

        left_thigh_text = "--"
        right_thigh_text = "--"
        left_shank_text = "--"
        right_shank_text = "--"

        left_knee_angle_text = "--"
        right_knee_angle_text = "--"

        displacement_text = (
            str(round(total_displacement_pixels, 1))
            + " px"
        )

        speed_text = (
            str(round(smoothed_speed_pixels, 1))
            + " px/s"
        )

        sway_status = "Not calibrated"
        calibration_status = "Waiting for person"
        legs_status = "Not visible"


        # =================================================
        # DETECT BODY POSE
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

            # Landmarks 0–10 are head/face landmarks.
            # Landmarks 11–32 are body landmarks.
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


            # Draw limb segments
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
            # CALCULATE WAIST CENTER
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
                # CALCULATE WAIST SPEED AND DISPLACEMENT
                # =================================================

                current_time = time.monotonic()


                if (
                    previous_waist_center is not None
                    and previous_waist_time is not None
                ):

                    elapsed_time = (
                        current_time
                        - previous_waist_time
                    )

                    if elapsed_time > 0:

                        frame_distance_pixels = (
                            pixel_distance(
                                previous_waist_center,
                                waist_center
                            )
                        )

                        current_speed_pixels = (
                            frame_distance_pixels
                            / elapsed_time
                        )

                        speed_history.append(
                            current_speed_pixels
                        )

                        smoothed_speed_pixels = (
                            0.8
                            * smoothed_speed_pixels
                            + 0.2
                            * np.mean(speed_history)
                        )

                        total_displacement_pixels += (
                            frame_distance_pixels
                        )


                previous_waist_center = waist_center
                previous_waist_time = current_time


                # =================================================
                # TEN-SECOND CALIBRATION
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

                        activity_label = (
                            "CALIBRATING"
                        )

                        calibration_status = (
                            "Stand still"
                        )

                        sway_status = (
                            "Box begins in "
                            + str(
                                round(
                                    calibration_remaining,
                                    1
                                )
                            )
                            + " s"
                        )


                    else:

                        # Average the waist position over the
                        # full ten-second calibration period
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

                        activity_label = "STANDING"
                        calibration_status = "Complete"
                        sway_status = "LOW SWAY"

                        print(
                            "Ten-second calibration complete."
                        )


                # =================================================
                # ACTIVITY CLASSIFICATION
                # =================================================

                if calibration_complete:

                    activity_label = classify_activity(
                        smoothed_speed_pixels
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


                    # Box center
                    cv2.circle(
                        frame,
                        sway_box_center,
                        6,
                        CYAN,
                        -1
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


                    # Coordinates of hips, knees, and ankles
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

                    legs_status = "Not visible"


        else:

            activity_label = "NO PERSON DETECTED"
            calibration_status = "No pose detected"


        # =========================================================
        # CAMERA-WINDOW INSTRUCTIONS
        # =========================================================

        draw_text(
            frame,
            "Press Q to quit | Press R to recalibrate",
            (20, 40),
            scale=0.7
        )

        draw_text(
            frame,
            "Red dots = body joints | Green lines = limbs",
            (20, 75),
            scale=0.6
        )

        draw_text(
            frame,
            "Purple dot = waist | Box = sway region",
            (20, 105),
            scale=0.6
        )


        # =========================================================
        # CREATE ALWAYS-VISIBLE SIDEBAR
        # =========================================================

        sidebar = np.zeros(
            (height, SIDEBAR_WIDTH, 3),
            dtype=np.uint8
        )

        sidebar[:] = (35, 35, 35)


        # Sidebar title
        sidebar_text(
            sidebar,
            "POSE MEASUREMENTS",
            40,
            color=CYAN,
            scale=0.75
        )

        cv2.line(
            sidebar,
            (20, 55),
            (SIDEBAR_WIDTH - 20, 55),
            GRAY,
            2
        )


        # Activity information
        sidebar_text(
            sidebar,
            "ACTIVITY",
            90,
            color=YELLOW,
            scale=0.65
        )

        sidebar_text(
            sidebar,
            activity_label,
            120,
            color=GREEN,
            scale=0.8
        )

        sidebar_text(
            sidebar,
            "Speed: " + speed_text,
            150
        )


        # Calibration information
        sidebar_text(
            sidebar,
            "CALIBRATION",
            200,
            color=YELLOW,
            scale=0.65
        )

        sidebar_text(
            sidebar,
            calibration_status,
            230
        )

        sidebar_text(
            sidebar,
            sway_status,
            260,
            color=GREEN if sway_status == "LOW SWAY"
            else RED if sway_status == "HIGH SWAY"
            else YELLOW
        )


        # Sway-box information
        sidebar_text(
            sidebar,
            "SWAY BOX",
            310,
            color=YELLOW,
            scale=0.65
        )

        sidebar_text(
            sidebar,
            "Width: "
            + str(BOX_WIDTH_PIXELS)
            + " px",
            340
        )

        sidebar_text(
            sidebar,
            "Height: "
            + str(BOX_HEIGHT_PIXELS)
            + " px",
            370
        )


        # Displacement
        sidebar_text(
            sidebar,
            "DISPLACEMENT",
            420,
            color=YELLOW,
            scale=0.65
        )

        sidebar_text(
            sidebar,
            "Waist path:",
            450
        )

        sidebar_text(
            sidebar,
            displacement_text,
            480,
            color=PURPLE,
            scale=0.75
        )


        # Segment lengths
        sidebar_text(
            sidebar,
            "SEGMENT LENGTHS",
            535,
            color=YELLOW,
            scale=0.65
        )

        sidebar_text(
            sidebar,
            "Left thigh: " + left_thigh_text,
            565
        )

        sidebar_text(
            sidebar,
            "Right thigh: " + right_thigh_text,
            595
        )

        sidebar_text(
            sidebar,
            "Left shank: " + left_shank_text,
            625
        )

        sidebar_text(
            sidebar,
            "Right shank: " + right_shank_text,
            655
        )


        # Knee angles
        sidebar_text(
            sidebar,
            "KNEE ANGLES",
            705,
            color=YELLOW,
            scale=0.65
        )

        sidebar_text(
            sidebar,
            "Left: " + left_knee_angle_text,
            735
        )

        sidebar_text(
            sidebar,
            "Right: " + right_knee_angle_text,
            765
        )


        # Leg visibility note
        sidebar_text(
            sidebar,
            "Legs: " + legs_status,
            815,
            color=GREEN if legs_status == "Visible"
            else RED,
            scale=0.6
        )

        sidebar_text(
            sidebar,
            "Lengths shown in pixels",
            850,
            color=CYAN,
            scale=0.55
        )


        # Combine camera view and sidebar
        combined_view = np.hstack(
            (frame, sidebar)
        )

        cv2.imshow(
            "Pose and Motion Tracker",
            combined_view
        )


        # =========================================================
        # KEYBOARD CONTROLS
        # =========================================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


        elif key == ord("r"):

            # Reset ten-second calibration
            calibration_start_time = None
            calibration_samples = []
            calibration_complete = False
            sway_box_center = None

            # Reset displacement and movement
            previous_waist_center = None
            previous_waist_time = None
            total_displacement_pixels = 0.0
            movement_trail = []

            # Reset activity
            speed_history.clear()
            smoothed_speed_pixels = 0.0
            activity_label = "WAITING"

            print(
                "Calibration reset. "
                "Stand still for 10 seconds."
            )


# =========================================================
# CLOSE PROGRAM
# =========================================================

camera.release()
cv2.destroyAllWindows()
