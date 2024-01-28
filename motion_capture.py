import cv2
import numpy as np
import time
import datetime
import math
import random
from PIL import ImageFont, ImageDraw, Image

def detect_motion(previous_frame, current_frame):
    diff = cv2.absdiff(previous_frame, current_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 100, 255, cv2.THRESH_BINARY) # Change Thresh to increase/decrease intensity
    kernel = np.ones((5,5),np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations = 2)
    return dilated

def rotoscope_frame(frame, mask):
    return cv2.bitwise_and(frame, frame, mask=mask)

def draw_face_rectangles(frame, rectangle_positions):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        rectangle_positions.append((x, y, w, h))
        if len(rectangle_positions) > 10:
            rectangle_positions.pop(0)

    for position in rectangle_positions:
        x, y, w, h = position
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'ERROR IDENTITY TRACKING FAILED', (x + w - 40, y + h - 90), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 1)
    return frame


cap = cv2.VideoCapture(0) # Change 0 to your video file path to upload videos to the program
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Initialize VideoWriter for MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec compatible with MP4
output_video = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

_, prev_frame = cap.read()
accumulated_motion = np.zeros_like(prev_frame)
rectangle_positions = []  # List to store positions of rectangles

# Initialize a variable to track the time for screenshots
last_screenshot_time = time.time()


blink_on = True
blink_interval = 1.0
last_blink_time = time.time() - blink_interval


def draw_centered_text(image, text, font, text_color, stroke_color, stroke_width):
    draw = ImageDraw.Draw(image)
    text_width, text_height = draw.textsize(text, font=font)
    x = (image.width - text_width) // 2
    y = (image.height - text_height) // 2

    draw.text((x-stroke_width, y), text, font=font, fill=stroke_color)
    draw.text((x+stroke_width, y), text, font=font, fill=stroke_color)
    draw.text((x, y-stroke_width), text, font=font, fill=stroke_color)
    draw.text((x, y+stroke_width), text, font=font, fill=stroke_color)

    draw.text((x, y), text, font=font, fill=text_color)


def draw_clock_with_stroke(frame):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    font_scale = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    white_color = (255, 255, 255)
    black_color = (0, 0, 0)

    bottom_left_corner = (10, frame.shape[0] - 20)

    offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for x_offset, y_offset in offsets:
        cv2.putText(frame, current_time,
                    (bottom_left_corner[0] + x_offset, bottom_left_corner[1] + y_offset),
                    font, font_scale, black_color, thickness)

    cv2.putText(frame, current_time, bottom_left_corner, font, font_scale, white_color, thickness)

def draw_scrolling_text(frame, text, font_scale, color, thickness, start_frame):

    height, width = frame.shape[:2]


    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    x = width - ((frame_count - start_frame) % (width + text_width))

    y = height - 70

    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

frame_count = 0
start_frame = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect motion
    motion_mask = detect_motion(prev_frame, frame)
    prev_frame = frame.copy()

    # Rotoscope current frame
    rotoscoped = rotoscope_frame(frame, motion_mask)

    # Overlay new motion onto the accumulated motion
    # Wherever the rotoscoped frame is non-zero, it will overwrite the accumulated frame
    accumulated_motion[rotoscoped != 0] = rotoscoped[rotoscoped != 0]

    motion_tracked_with_faces = draw_face_rectangles(accumulated_motion.copy(), rectangle_positions)

    cam_text = 'CAM 22'
    cam_position = (10, 55)
    cv2.putText(motion_tracked_with_faces, cam_text, cam_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)  # Black stroke
    cv2.putText(motion_tracked_with_faces, cam_text, cam_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # White text

    draw_clock_with_stroke(motion_tracked_with_faces)

    recording_text = "Recording"
    recording_font_scale = 0.8 
    recording_x = 1650
    recording_y = 1050

    circle_radius = 12
    circle_center = (recording_x + 110, recording_y - 10)
    overlay = motion_tracked_with_faces.copy()
    cv2.circle(overlay, circle_center, circle_radius, (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.5, motion_tracked_with_faces, 0.5, 0, motion_tracked_with_faces)

    cv2.putText(motion_tracked_with_faces, recording_text, (recording_x + 130, recording_y), cv2.FONT_HERSHEY_SIMPLEX, recording_font_scale, (0, 0, 0), 4)  # Black stroke
    cv2.putText(motion_tracked_with_faces, recording_text, (recording_x + 130, recording_y), cv2.FONT_HERSHEY_SIMPLEX, recording_font_scale, (255, 255, 255), 2)  # White text


    if time.time() - last_blink_time > blink_interval:
        blink_on = not blink_on
        last_blink_time = time.time()

    if blink_on:
        identity_tracking_text = "IDENTITY TRACKING IN PROGRESS"
        text_bottom_left_x = 870
        text_bottom_left_y = frame.shape[0] - 1030
        cv2.putText(motion_tracked_with_faces, identity_tracking_text, (text_bottom_left_x, text_bottom_left_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8)  # Black stroke
        cv2.putText(motion_tracked_with_faces, identity_tracking_text, (text_bottom_left_x, text_bottom_left_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)  # Green text

    # Check if 30 seconds have passed to take a screenshot
    if time.time() - last_screenshot_time > 30:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f'screenshot_{timestamp}.png', motion_tracked_with_faces)
        last_screenshot_time = time.time()

       # Write the processed frame to the video file
    output_video.write(motion_tracked_with_faces)


    frame_count += 8
    draw_scrolling_text(motion_tracked_with_faces, "PRIVACY AS YOU KNEW IT DOESN'T EXIST ANYMORE", 1, (0, 255, 0), 2, start_frame)

    cv2.imshow('Permanent Motion Trails with Face Detection & Text', motion_tracked_with_faces)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_video.release()  # Make sure to release the video writer object
cv2.destroyAllWindows()