#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'depthai_hand_tracker'))
from HandTracker import HandTracker
from HandTrackerRenderer import HandTrackerRenderer
import argparse
import time
import matplotlib.pyplot as plt
from collections import deque
from mpl_toolkits.mplot3d import Axes3D 

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--edge', action="store_true",
                    help="Use Edge mode (postprocessing runs on the device)")
parser_tracker = parser.add_argument_group("Tracker arguments")
parser_tracker.add_argument('-i', '--input', type=str, 
                    help="Path to video or image file to use as input (if not specified, use OAK color camera)")
parser_tracker.add_argument("--pd_model", type=str,
                    help="Path to a blob file for palm detection model")
parser_tracker.add_argument('--no_lm', action="store_true", 
                    help="Only the palm detection model is run (no hand landmark model)")
parser_tracker.add_argument("--lm_model", type=str,
                    help="Landmark model 'full', 'lite', 'sparse' or path to a blob file")
parser_tracker.add_argument('--use_world_landmarks', action="store_true", 
                    help="Fetch landmark 3D coordinates in meter")
parser_tracker.add_argument('-s', '--solo', action="store_true", 
                    help="Solo mode: detect one hand max. If not used, detect 2 hands max (Duo mode)")                    
parser_tracker.add_argument('-xyz', "--xyz", action="store_true", 
                    help="Enable spatial location measure of palm centers")
parser_tracker.add_argument('-g', '--gesture', action="store_true", 
                    help="Enable gesture recognition")
parser_tracker.add_argument('-c', '--crop', action="store_true", 
                    help="Center crop frames to a square shape")
parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument("-r", "--resolution", choices=['full', 'ultra'], default='full',
                    help="Sensor resolution: 'full' (1920x1080) or 'ultra' (3840x2160) (default=%(default)s)")
parser_tracker.add_argument('--internal_frame_height', type=int,                                                                                 
                    help="Internal color camera frame height in pixels")   
parser_tracker.add_argument("-lh", "--use_last_handedness", action="store_true",
                    help="Use last inferred handedness. Otherwise use handedness average (more robust)")                            
parser_tracker.add_argument('--single_hand_tolerance_thresh', type=int, default=10,
                    help="(Duo mode only) Number of frames after only one hand is detected before calling palm detection (default=%(default)s)")
parser_tracker.add_argument('--dont_force_same_image', action="store_true",
                    help="(Edge Duo mode only) Don't force the use the same image when inferring the landmarks of the 2 hands (slower but skeleton less shifted)")
parser_tracker.add_argument('-lmt', '--lm_nb_threads', type=int, choices=[1,2], default=2, 
                    help="Number of the landmark model inference threads (default=%(default)i)")  
parser_tracker.add_argument('-t', '--trace', type=int, nargs="?", const=1, default=0, 
                    help="Print some debug infos. The type of info depends on the optional argument.")                
parser_renderer = parser.add_argument_group("Renderer arguments")
parser_renderer.add_argument('-o', '--output', 
                    help="Path to output video file")
args = parser.parse_args()
dargs = vars(args)
tracker_args = {a:dargs[a] for a in ['pd_model', 'lm_model', 'internal_fps', 'internal_frame_height'] if dargs[a] is not None}

if args.edge:
    from HandTrackerEdge import HandTracker
    tracker_args['use_same_image'] = not args.dont_force_same_image
else:
    from HandTracker import HandTracker


tracker = HandTracker(
        input_src=args.input, 
        use_lm= not args.no_lm, 
        use_world_landmarks=args.use_world_landmarks,
        use_gesture=args.gesture,
        xyz=args.xyz,
        solo=args.solo,
        crop=args.crop,
        resolution=args.resolution,
        stats=True,
        trace=args.trace,
        use_handedness_average=not args.use_last_handedness,
        single_hand_tolerance_thresh=args.single_hand_tolerance_thresh,
        lm_nb_threads=args.lm_nb_threads,
        **tracker_args
        )

renderer = HandTrackerRenderer(
        tracker=tracker,
        output=args.output)

# 1. Initialize fixed-size windows for 3D coordinates (Spatial Data)
MAX_SAMPLES = 100
x_data = deque(maxlen=MAX_SAMPLES)
y_data = deque(maxlen=MAX_SAMPLES)
z_data = deque(maxlen=MAX_SAMPLES)

# 2. Initialize rolling windows for Velocity and Time
v_time_data = deque(maxlen=MAX_SAMPLES)
x_vel_data = deque(maxlen=MAX_SAMPLES)
y_vel_data = deque(maxlen=MAX_SAMPLES)
z_vel_data = deque(maxlen=MAX_SAMPLES)

plt.ion()

# --- PLOT SETUP: Window 1 (3D Position) ---
fig_3d = plt.figure(1)
ax_3d = fig_3d.add_subplot(projection='3d')
path_line, = ax_3d.plot([], [], [], 'b-', label='Trailing Path')
current_pos, = ax_3d.plot([], [], [], 'ro', markersize=8, label='Object')
ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')
ax_3d.legend()

BOX_MIN, BOX_MAX = -500, 500
ax_3d.set_xlim(BOX_MIN, BOX_MAX)
ax_3d.set_ylim(BOX_MIN, BOX_MAX)
ax_3d.set_zlim(500, 1500)

# --- PLOT SETUP: Window 2 (2D Live Velocity) ---
fig_vel, ax_vel = plt.subplots(num=2)
vx_line, = ax_vel.plot([], [], 'r-', label='Filtered X Vel')
vy_line, = ax_vel.plot([], [], 'g-', label='Filtered Y Vel')
vz_line, = ax_vel.plot([], [], 'b-', label='Filtered Z Vel')
ax_vel.set_xlabel('Time (s)')
ax_vel.set_ylabel('Velocity (units/s)')
ax_vel.set_ylim(-2000, 2000) 
ax_vel.legend()
ax_vel.grid(True)

# --- PLOT SETUP: Window 2 (2D Live Pos) ---
fig_pos, ax_pos = plt.subplots(num=3)
px_line, = ax_pos.plot([], [], 'r-', label='Filtered X Pos')
py_line, = ax_pos.plot([], [], 'g-', label='Filtered Y Pos')
pz_line, = ax_pos.plot([], [], 'b-', label='Filtered Z Pos')
ax_pos.set_xlabel('Time (s)')
ax_pos.set_ylabel('Pos (units)')
ax_pos.set_ylim(-1000, 1000) 
ax_pos.legend()
ax_pos.grid(True)

pinch_threshold = 30
start_time = time.perf_counter()

# --- ALPHA-BETA FILTER INITIALIZATION ---
# Tuning constants: adjust these to balance responsiveness vs. smoothness
ALPHA = 0.3   
BETA = 0.2    

# Track filter states across iterations (X, Y, Z)
filter_pos = None  # Will hold np.array([x, y, z])
filter_vel = np.array([0.0, 0.0, 0.0])
last_time = None
# ----------------------------------------

try:
    while True:
        frame, hands, bag = tracker.next_frame()
        if frame is None:
            break

        frame = renderer.draw(frame, hands, bag)
        for hand in hands:
            thumb_tip = np.array(hand.landmarks[4])
            index_tip = np.array(hand.landmarks[8])
            
            current_time = time.perf_counter() - start_time
            hand_pos = np.array(hand.xyz) # Make sure this is a numpy array
            distance = np.linalg.norm(thumb_tip - index_tip)
            print(distance)
            is_pinching = distance < pinch_threshold

            if is_pinching:
                # --- RUN ALPHA-BETA FILTER ---
                if filter_pos is None or last_time is None:
                    # First sample: Initialize positions with raw data, velocities at zero
                    filter_pos = hand_pos.copy()
                    filter_vel = np.array([0.0, 0.0, 0.0])
                    dt = 0.001 # Small default step
                else:
                    dt = current_time - last_time
                    # if dt <= 0:  # Safety check for fast frame updates
                    #     dt = 0.001
                    
                    # 1. State Prediction Step
                    pred_pos = filter_pos + (filter_vel * dt)
                    
                    # 2. Residual Calculation Step
                    residual = hand_pos - pred_pos
                    
                    # 3. State Correction Step
                    filter_pos = pred_pos + (ALPHA * residual)
                    filter_vel = filter_vel + ((BETA / dt) * residual)
                
                last_time = current_time
                # ------------------------------

                # Append Filtered 3D Positions to queues
                x_data.append(filter_pos[0])
                y_data.append(filter_pos[1])
                z_data.append(filter_pos[2])
                
                # Append Filtered Velocities and Times to queues
                v_time_data.append(current_time)
                x_vel_data.append(filter_vel[0])
                y_vel_data.append(filter_vel[1])
                z_vel_data.append(filter_vel[2])

                # Update 3D Canvas
                path_line.set_data(list(x_data), list(y_data))
                path_line.set_3d_properties(list(z_data))
                current_pos.set_data([filter_pos[0]], [filter_pos[1]])
                current_pos.set_3d_properties([filter_pos[2]])

                # Update 2D Pos Canvas
                px_line.set_data(list(v_time_data), list(x_data))
                py_line.set_data(list(v_time_data), list(y_data))
                pz_line.set_data(list(v_time_data), list(z_data))
                if len(v_time_data) > 0:
                    ax_pos.set_xlim(v_time_data[0], v_time_data[-1] + 0.5)
                # Update 2D Velocity Canvas
                vx_line.set_data(list(v_time_data), list(x_vel_data))
                vy_line.set_data(list(v_time_data), list(y_vel_data))
                vz_line.set_data(list(v_time_data), list(z_vel_data))
                
                if len(v_time_data) > 0:
                    ax_vel.set_xlim(v_time_data[0], v_time_data[-1] + 0.5)
            else:
                # If they stop pinching, reset the filter memory for the next stroke
                filter_pos = None
                last_time = None

        fig_3d.canvas.draw()
        fig_3d.canvas.flush_events()
        fig_vel.canvas.draw()
        fig_vel.canvas.flush_events()
        fig_pos.canvas.draw()
        fig_pos.canvas.flush_events()
        time.sleep(0.001)
        cv2.imshow("hand tracker", frame)
        if cv2.waitKey(1) == 27:  # Esc to quit
            break

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cv2.destroyAllWindows()
    plt.close('all')
    plt.ioff()
