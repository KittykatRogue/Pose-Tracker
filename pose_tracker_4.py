import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
import time
import math


# =========================================================
# SETTINGS
# =========================================================

MARKER_ID = 0
MARKER_SIZE_METERS = 0.10

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
    """
    Calculates the distance between two image points in pixels.
    """

    return math.dist(point_a, point_b)


def draw_text(
    frame,
    text,
    location,
    color=(255, 255, 255),
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
# SET UP ARUCO DETECTOR
# =========================================================

aruco_dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

try:
    aruco_parameters = cv2.aruco.DetectorParameters()

except AttributeError:
    aruco_parameters = cv2.aruco.DetectorParameters_create()


try:
    aruco_detector = cv2.aruco.ArucoDetector(
        aruco_dictionary,
        aruco_parameters
    )

except AttributeError:
    aruco_detector = None


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

# Request a higher resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not camera.isOpened():

    print("Could not open the webcam.")
    exit()


# =========================================================
# TRACKING VARIABLES
# =========================================================

previous_marker_center = None
total_displacement_meters = 0.0

movement_trail = []

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


        # Do not flip the image.
        # Flipping can prevent ArUco detection.
        height, width, _ = frame.shape


        # =================================================
        # DETECT ARUCO MARKER
        # =================================================

        if aruco_detector is not None:

            marker_corners, marker_ids, rejected = (
                aruco_detector.detectMarkers(frame)
            )

        else:

            marker_corners, marker_ids, rejected = (
                cv2.aruco.detectMarkers(
                    frame,
                    aruco_dictionary,
                    parameters=aruco_parameters
                )
            )


        marker_detected = False
        meters_per_pixel = None
        current_marker_center = None


        if marker_ids is not None:

            for i, detected_id in enumerate(
                marker_ids.flatten()
            ):

                if detected_id == MARKER_ID:

                    marker_detected = True

                    corners = marker_corners[i][0]

                    # Draw yellow marker outline
                    cv2.polylines(
                        frame,
                        [corners.astype(np.int32)],
                        True,
                        (0, 255, 255),
                        3
                    )

                    # Marker center
                    marker_x = int(
                        np.mean(corners[:, 0])
                    )

                    marker_y = int(
                        np.mean(corners[:, 1])
                    )

                    current_marker_center = (
                        marker_x,
                        marker_y
                    )

                    # Estimate marker width in pixels
                    top_width = np.linalg.norm(
                        corners[0] - corners[1]
                    )

                    bottom_width = np.linalg.norm(
                        corners[2] - corners[3]
                    )

                    marker_width_pixels = (
                        top_width + bottom_width
                    ) / 2


                    if marker_width_pixels > 0:

                        meters_per_pixel = (
                            MARKER_SIZE_METERS
                            / marker_width_pixels
                        )


                    # Draw pink marker center
                    cv2.circle(
                        frame,
                        current_marker_center,
                        9,
                        (255, 0, 255),
                        -1
                    )

                    draw_text(
                        frame,
                        "ArUco ID 0 detected",
                        (marker_x + 15, marker_y),
                        color=(0, 255, 255)
                    )


                    # Calculate marker displacement
                    if (
                        previous_marker_center is not None
                        and meters_per_pixel is not None
                    ):

                        frame_displacement_pixels = (
                            pixel_distance(
                                previous_marker_center,
                                current_marker_center
                            )
                        )

                        frame_displacement_meters = (
                            frame_displacement_pixels
                            * meters_per_pixel
                        )

                        total_displacement_meters += (
                            frame_displacement_meters
                        )


                    previous_marker_center = (
                        current_marker_center
                    )


                    # Store movement trail
                    movement_trail.append(
                        current_marker_center
                    )

                    if len(movement_trail) > 150:

                        movement_trail.pop(0)


                    # Draw movement trail
                    for j in range(1, len(movement_trail)):

                        cv2.line(
                            frame,
                            movement_trail[j - 1],
                            movement_trail[j],
                            (255, 0, 255),
                            3
                        )


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


        legs_are_visible = False


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
            # DRAW ONLY BODY JOINTS
            # =================================================

            # Indices 0–10 include the face and head.
            # Indices 11–32 are shoulders, arms, hips, legs,
            # and feet.
            body_joint_indices = list(range(11, 33))

            for index in body_joint_indices:

                x, y, visibility = points[index]

                if visibility > 0.5:

                    cv2.circle(
                        frame,
                        (x, y),
                        6,
                        (0, 0, 255),
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
                        (0, 255, 0),
                        3
                    )


            # =================================================
            # DRAW PURPLE WAIST DOT
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

                # Purple waist dot
                cv2.circle(
                    frame,
                    waist_center,
                    11,
                    (255, 0, 255),
                    -1
                )

                draw_text(
                    frame,
                    "Waist",
                    (waist_x + 15, waist_y),
                    color=(255, 0, 255)
                )


            # =================================================
            # CHECK WHETHER LEG JOINTS ARE VISIBLE
            # =================================================

            required_leg_joints = [
                23, 24, 25, 26, 27, 28
            ]

            legs_are_visible = all(
                points[index][2] > 0.5
                for index in required_leg_joints
            )


            # =================================================
            # CALCULATE MEASUREMENTS
            # =================================================

            if (
                marker_detected
                and meters_per_pixel is not None
                and legs_are_visible
            ):

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


                # Convert segment lengths to meters
                left_thigh_meters = (
                    left_thigh_pixels
                    * meters_per_pixel
                )

                right_thigh_meters = (
                    right_thigh_pixels
                    * meters_per_pixel
                )

                left_shank_meters = (
                    left_shank_pixels
                    * meters_per_pixel
                )

                right_shank_meters = (
                    right_shank_pixels
                    * meters_per_pixel
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


                # =================================================
                # DISPLAY MEASUREMENTS
                # =================================================

                measurement_x = 20
                measurement_y = 155

                draw_text(
                    frame,
                    "METRICS ACTIVE",
                    (measurement_x, measurement_y),
                    color=(0, 255, 0),
                    scale=0.7
                )

                draw_text(
                    frame,
                    "Left thigh: "
                    + str(round(left_thigh_meters, 3))
                    + " m",
                    (measurement_x, measurement_y + 32)
                )

                draw_text(
                    frame,
                    "Right thigh: "
                    + str(round(right_thigh_meters, 3))
                    + " m",
                    (measurement_x, measurement_y + 62)
                )

                draw_text(
                    frame,
                    "Left shank: "
                    + str(round(left_shank_meters, 3))
                    + " m",
                    (measurement_x, measurement_y + 92)
                )

                draw_text(
                    frame,
                    "Right shank: "
                    + str(round(right_shank_meters, 3))
                    + " m",
                    (measurement_x, measurement_y + 122)
                )

                draw_text(
                    frame,
                    "Left knee angle: "
                    + str(round(left_knee_angle, 1))
                    + " deg",
                    (measurement_x, measurement_y + 152)
                )

                draw_text(
                    frame,
                    "Right knee angle: "
                    + str(round(right_knee_angle, 1))
                    + " deg",
                    (measurement_x, measurement_y + 182)
                )

                draw_text(
                    frame,
                    "Displacement: "
                    + str(round(total_displacement_meters, 3))
                    + " m",
                    (measurement_x, measurement_y + 212),
                    color=(255, 0, 255),
                    scale=0.7
                )


            elif marker_detected and not legs_are_visible:

                draw_text(
                    frame,
                    "Marker detected, but legs are not visible",
                    (20, 155),
                    color=(0, 255, 255)
                )


            elif not marker_detected:

                draw_text(
                    frame,
                    "Show ArUco marker to activate measurements",
                    (20, 155),
                    color=(0, 255, 255)
                )


        # =================================================
        # GENERAL INFORMATION
        # =================================================

        draw_text(
            frame,
            "Press Q to quit",
            (20, 40),
            scale=0.75
        )

        draw_text(
            frame,
            "Red dots = body joints | Green lines = limbs",
            (20, 75),
            scale=0.6
        )

        draw_text(
            frame,
            "Purple dot = waist | Yellow box = ArUco marker",
            (20, 105),
            scale=0.6
        )

        cv2.imshow(
            "Pose, Measurements, and Displacement",
            frame
        )


        # Press Q to close
        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# =========================================================
# CLOSE CAMERA
# =========================================================

camera.release()
cv2.destroyAllWindows()
