import cv2
import mediapipe as mp
import time # Needed for timestamp
import os

# Suppress TensorFlow and MediaPipe logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging # Shows only errors in TF
import logging
logging.basicConfig(level=logging.ERROR)  # Set logging level to ERROR

# --- Import MediaPipe Task libraries ---
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# --- Global variable to store the latest results ---
latest_result = None

# Define GestureRecognizerResult before using it
from mediapipe.tasks.python.vision import GestureRecognizerResult

# --- Performance metrics variables ---
performance_stats = {
    'callback_count': 0,
    'processing_times': [],
    'last_processed_time': 0
}

# --- Callback function to handle results ---
def save_result_callback(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result, performance_stats
    latest_result = result
    
    # Track callback performance
    current_time = time.time() * 1000
    if performance_stats['last_processed_time'] > 0:
        processing_time = current_time - performance_stats['last_processed_time']
        performance_stats['processing_times'].append(processing_time)
        # Keep only last 30 measurements
        if len(performance_stats['processing_times']) > 30:
            performance_stats['processing_times'].pop(0)
    
    performance_stats['last_processed_time'] = current_time
    performance_stats['callback_count'] += 1


model_path = r"D:\projects\motion tracking\gesture_recognizer.task" # Path to the model file

# Define task-specific options and model settings
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Configure base options with performance optimizations
base_options = BaseOptions(model_asset_path=model_path)

# Set up GPU or CPU optimization based on availability
try:
    # Set environment variables for better performance
    os.environ['TFLITE_GPU_ALLOW_PRECISION_LOSS'] = '1'  # Allow lower precision for speed
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow memory growth for GPU usage
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable Intel MKL-DNN optimizations
    
    # Define number of threads for CPU execution
    import multiprocessing
    num_cpu = multiprocessing.cpu_count()
    num_threads = max(1, num_cpu - 1)  # Leave one CPU free for system tasks
    
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    
    # XNNPACK optimization (may help with CPU performance)
    os.environ['TFLITE_XNNPACK_MAX_NUM_THREADS'] = str(num_threads)
    
    print(f"Performance optimizations applied using {num_threads} threads")
    
    # Add more specific MediaPipe optimization settings
    # Some of these may not be supported in your version but won't cause errors
    os.environ['MEDIAPIPE_DISABLE_GPU'] = '0'  # Try to enable GPU if available
    os.environ['MEDIAPIPE_GPU_INFERENCE'] = '1'
    
except Exception as e:
    print(f"Could not apply all optimizations: {e}")

# Create a prototype with customized options
options = GestureRecognizerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,  # Detect up to two hands
    result_callback=save_result_callback,
    # Adjust these thresholds for better performance - lower values = faster but less accurate
    min_tracking_confidence=0.3,  # Default: 0.5
    min_hand_presence_confidence=0.3,  # Default: 0.5
    min_hand_detection_confidence=0.3)  # Default: 0.5

# Print information about current configuration
print("\nGesture Recognizer Settings:")
print(f"- Processing in LIVE_STREAM mode")
print(f"- Tracking up to 2 hands")
print(f"- Using lower confidence thresholds (0.3) for better performance")
print("- Use 'l' to toggle resolution, '+'/'-' to adjust frame skip\n")


# Use a try-except block for better error handling if the model file is missing
try:
    recognizer = vision.GestureRecognizer.create_from_options(options)
except Exception as e:
    print(f"Error creating GestureRecognizer: {e}")
    print(f"Ensure the model file '{model_path}' exists and the path is correct.")
    exit()

# Initialize video capture from the default webcam (index 0)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set frame dimensions (optional, MediaPipe handles different sizes)
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

print("Starting MediaPipe gesture recognition. Press 'q' to quit.")
print("Controls: '+'/'-' to adjust frame skip, 'r' to reset stats, 'q' to quit")

frame_timestamp_ms = 0
frame_count = 0
last_fps_time = time.time()
fps = 0
processing_time = 0
frame_skip = 2  # Start with processing every 2nd frame for better performance
start_resolution = (frame_width, frame_height)  # Original resolution
lowres_mode = True  # Start in low resolution mode for better performance

# Pre-allocate image buffer to avoid memory allocations in the loop
# This helps reduce garbage collection pauses
buffer_frame = None

while True:
    loop_start_time = time.time()
    
    # Read a frame from the webcam
    ret, frame = cap.read()

    # If frame reading was unsuccessful, break the loop
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Always flip the frame for selfie view (this is cheap)
    frame = cv2.flip(frame, 1)
    
    # Process only every n-th frame to reduce CPU/GPU load
    process_this_frame = (frame_count % frame_skip == 0)
    
    if process_this_frame:
        # Use lower resolution in lowres_mode for better performance
        if lowres_mode:
            # Use half resolution
            process_width = frame_width // 2
            process_height = frame_height // 2
            process_frame = cv2.resize(frame, (process_width, process_height))
        else:
            process_width, process_height = frame_width, frame_height
            process_frame = frame
            
        # --- MediaPipe Gesture Recognition Logic ---
        # 1. Convert the BGR frame to RGB then to MediaPipe Image format
        # Avoid creating new arrays by reusing the buffer when possible
        rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 2. Get current timestamp for async call
        frame_timestamp_ms = int(time.time() * 1000)

        # 3. Send frame to recognizer asynchronously (Intensive processing)
        recognizer.recognize_async(mp_image, frame_timestamp_ms)
    
    # FPS calculation
    frame_count += 1
    if frame_count >= 30:  # Update FPS every 30 frames
        current_time = time.time()
        fps = frame_count / (current_time - last_fps_time)
        frame_count = 0
        last_fps_time = current_time
    
    # Calculate per-frame processing time
    processing_time = (time.time() - loop_start_time) * 1000  # in ms

    # 4. Draw results (if available from callback)
    current_result = latest_result # Copy to avoid race conditions
    if current_result:
        # Loop through detected hands
        for i in range(len(current_result.gestures)):
            # Draw hand landmarks
            if current_result.hand_landmarks:
                hand_landmarks = current_result.hand_landmarks[i]
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks_proto,
                    solutions.hands.HAND_CONNECTIONS,
                    solutions.drawing_styles.get_default_hand_landmarks_style(),
                    solutions.drawing_styles.get_default_hand_connections_style())

            # Display recognized gesture
            if current_result.gestures and current_result.gestures[i]: # Check if gesture list is not empty for this hand
                gesture = current_result.gestures[i]
                if gesture: # Check if the gesture list itself is not empty
                    category_name = gesture[0].category_name # Get the top gesture
                    score = round(gesture[0].score, 2)
                    # Find bounding box to position text
                    x_min, y_min = frame_width, frame_height
                    x_max, y_max = 0, 0
                    if current_result.hand_landmarks and current_result.hand_landmarks[i]:
                        for landmark in current_result.hand_landmarks[i]:
                            x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                            if x < x_min: x_min = x
                            if y < y_min: y_min = y
                            if x > x_max: x_max = x
                            if y > y_max: y_max = y

                    text_x = x_min
                    text_y = y_min - 10 # Position above the hand
                    if text_y < 10: text_y = y_max + 20 # If too high, position below

                    cv2.putText(frame, f"{category_name} ({score})",
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # Create a semi-transparent overlay for metrics
    metrics_overlay = frame.copy()
    cv2.rectangle(metrics_overlay, (5, 5), (280, 180), (0, 0, 0), -1)
    cv2.addWeighted(metrics_overlay, 0.6, frame, 0.4, 0, frame)
    
    # Display performance metrics with better visibility
    if fps > 0:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Process: {processing_time:.1f}ms", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Skip: {frame_skip} frames", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Show resolution mode
        resolution_text = "Low Res" if lowres_mode else "Full Res"
        cv2.putText(frame, f"Mode: {resolution_text}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Add callback performance stats
        if performance_stats['processing_times']:
            avg_callback_time = sum(performance_stats['processing_times']) / len(performance_stats['processing_times'])
            cv2.putText(frame, f"Callback: {avg_callback_time:.1f}ms", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
    # Display the resulting frame
    cv2.imshow("MediaPipe Gesture Recognition", frame)

    # Check for keys - make the wait time adaptive based on FPS
    wait_time = max(1, int(1000/60))  # Cap at 60 FPS max for display
    key = cv2.waitKey(wait_time) & 0xFF
    
    # Enhanced controls:
    # q - quit
    # + - increase frame_skip (process fewer frames)
    # - - decrease frame_skip (process more frames)
    # r - reset performance stats
    # l - toggle resolution mode
    # h - display help
    if key == ord('q'):
        print("Exiting program.")
        break
    elif key == ord('+'):
        frame_skip = min(10, frame_skip + 1)
        print(f"Processing every {frame_skip} frames")
    elif key == ord('-'):
        frame_skip = max(1, frame_skip - 1)
        print(f"Processing every {frame_skip} frames")
    elif key == ord('r'):
        # Reset performance stats
        performance_stats['processing_times'] = []
        performance_stats['callback_count'] = 0
        print("Performance metrics reset")
    elif key == ord('l'):
        # Toggle resolution mode
        lowres_mode = not lowres_mode
        print(f"Resolution mode: {'Low' if lowres_mode else 'Full'}")
    elif key == ord('h'):
        # Display help
        print("\nControls:")
        print("  q - Quit the application")
        print("  + - Increase frame skip (process fewer frames)")
        print("  - - Decrease frame skip (process more frames)")
        print("  r - Reset performance statistics")
        print("  l - Toggle between low and full resolution")
        print("  h - Display this help message")

# Release the capture object, close recognizer, and destroy all windows
recognizer.close() # Close the recognizer
cap.release()
cv2.destroyAllWindows()
