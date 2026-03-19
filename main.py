import cv2
import numpy as np
import mediapipe as mp
import threading
import PIL.Image
from google import genai
import tensorflow as tf
import os
import time

# ============================================================
#  1. LOAD THE TRAINED CNN MODEL
# ============================================================
print("Loading CNN model, please wait...")
model = tf.keras.models.load_model('calculator_brain.h5')
print("Model loaded successfully.")

# ============================================================
#  2. GEMINI API CONFIGURATION (for complex expressions)
# ============================================================
API_KEY = "YOUR_API_KEY_HERE"
client = genai.Client(api_key=API_KEY)

# ============================================================
#  3. CAMERA & HAND TRACKING SETUP
# ============================================================
camera = cv2.VideoCapture(0, cv2.CAP_MSMF)
camera.set(3, 1280)
camera.set(4, 720)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Drawing canvas and state variables
canvas = np.zeros((720, 1280, 3), np.uint8)
prev_x, prev_y = 0, 0
display_result = ""

# ============================================================
#  4. CORRECTION SYSTEM – DATA STRUCTURES
# ============================================================
# Stores the last recognized segments: [(28x28_img, predicted_label, predicted_index, (x,y,w,h)), ...]
last_segments = []
last_binary_image = None  # Original binary image (needed for merge/split)

# Character -> dataset folder name mapping
CHAR_TO_FOLDER = {
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
    '+': 'plus', '-': 'minus', '*': 'mult', '/': 'divide'
}

# Key code -> character mapping (for cv2.waitKey)
KEY_TO_CHAR = {
    ord('0'): '0', ord('1'): '1', ord('2'): '2', ord('3'): '3', ord('4'): '4',
    ord('5'): '5', ord('6'): '6', ord('7'): '7', ord('8'): '8', ord('9'): '9',
    ord('+'): '+', ord('-'): '-', ord('*'): '*', ord('/'): '/',
    # Convenient letter shortcuts for operators
    ord('p'): '+',  # p = plus
    ord('m'): '-',  # m = minus
    ord('x'): '*',  # x = multiply
    ord('b'): '/',  # b = divide (bölme)
}

# Class index -> math symbol mapping (alphabetical folder order from Keras)
CLASS_MAP = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "/",  # divide
    11: "-",  # minus
    12: "*",  # mult
    13: "+"   # plus
}


# ============================================================
#  5. GEMINI API – ASYNC REQUEST
# ============================================================
def send_to_api(canvas_copy):
    """Send the drawn expression to Gemini API for evaluation."""
    global display_result
    try:
        cv2.imwrite("temp_expression.jpg", canvas_copy)
        img = PIL.Image.open("temp_expression.jpg")
        prompt = (
            "This image contains a handwritten math expression. "
            "Solve it and return only the numerical result."
        )
        response = client.models.generate_content(
            model='gemini-2.5-flash', contents=[prompt, img]
        )
        display_result = f"API Result: {response.text.strip()}"
    except Exception as e:
        display_result = "API Error"


# ============================================================
#  6. MAIN LOOP
# ============================================================
while True:
    success, frame = camera.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            h, w, c = frame.shape
            x1 = int(landmarks[8].x * w)
            y1 = int(landmarks[8].y * h)
            x2 = int(landmarks[12].x * w)
            y2 = int(landmarks[12].y * h)

            index_up = landmarks[8].y < landmarks[5].y
            middle_up = landmarks[12].y < landmarks[9].y

            # Drawing mode: only index finger raised
            if index_up and not middle_up:
                cv2.circle(frame, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x1, y1
                cv2.line(canvas, (prev_x, prev_y), (x1, y1), (255, 0, 255), 15)
                prev_x, prev_y = x1, y1

            # Selection mode: both index and middle fingers raised
            elif index_up and middle_up:
                prev_x, prev_y = 0, 0
                cv2.rectangle(frame, (x1, y1 - 25), (x2, y2 + 25),
                              (0, 0, 255), cv2.FILLED)
            else:
                prev_x, prev_y = 0, 0

    # Overlay the canvas onto the camera frame
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv_mask = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY_INV)
    inv_mask_bgr = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, inv_mask_bgr)
    frame = cv2.bitwise_or(frame, canvas)

    # Show current result on screen
    if display_result != "":
        cv2.putText(frame, display_result, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

    cv2.imshow("Hand Calculator - CNN + Gemini AI", frame)
    key = cv2.waitKey(1) & 0xFF

    # ----------------------------------------------------------
    #  [C] Clear the canvas
    # ----------------------------------------------------------
    if key == ord('c'):
        canvas = np.zeros((720, 1280, 3), np.uint8)
        display_result = ""

    # ----------------------------------------------------------
    #  [S] Send to Gemini API (for complex expressions)
    # ----------------------------------------------------------
    elif key == ord('s'):
        if display_result != "Computing...":
            display_result = "Computing..."
            api_thread = threading.Thread(
                target=send_to_api, args=(canvas.copy(),)
            )
            api_thread.start()

    # ----------------------------------------------------------
    #  [A] Analyze with local CNN model
    # ----------------------------------------------------------
    elif key == ord('a'):
        print("Running local CNN analysis...")

        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Apply erosion to separate touching characters
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)

        # Store binary image for merge/split in correction mode
        last_binary_image = binary.copy()

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            boxes = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # Keep shapes larger than 10px (avoids noise, keeps thin operators like '-')
                if w > 10 or h > 10:
                    boxes.append((x, y, w, h))

            if not boxes:
                display_result = "No valid drawing found!"
                print(display_result)
                continue

            # Sort bounding boxes left-to-right
            boxes = sorted(boxes, key=lambda box: box[0])

            equation = ""
            debug_images = []
            last_segments = []

            # Process each detected segment
            for (x, y, w, h) in boxes:
                cropped = binary[y:y + h, x:x + w]
                padding = 20
                padded = cv2.copyMakeBorder(
                    cropped, padding, padding, padding, padding,
                    cv2.BORDER_CONSTANT, value=0
                )

                # Invert colors: black background, white strokes -> white bg, black strokes
                padded = cv2.bitwise_not(padded)

                # Resize to 28x28 (model input size)
                resized = cv2.resize(padded, (28, 28))
                debug_images.append(resized)

                # Normalize and reshape for prediction
                normalized = resized / 255.0
                normalized = normalized.reshape(1, 28, 28, 1)

                prediction = model.predict(normalized, verbose=0)
                predicted_index = np.argmax(prediction)

                predicted_char = CLASS_MAP.get(predicted_index, "")
                equation += predicted_char

                # Store segment data for correction mode
                last_segments.append(
                    (resized.copy(), predicted_char, predicted_index, (x, y, w, h))
                )

            print(f"Detected equation: {equation}")

            # Show each segment side by side in a debug window
            if len(debug_images) > 0:
                combined = np.hstack(debug_images)
                scaled = cv2.resize(combined, (100 * len(debug_images), 100))
                cv2.imshow("Model's View", scaled)

            # Evaluate the detected expression
            try:
                result = eval(equation)
                display_result = f"{equation} = {result}"
                print(f"SUCCESS: {display_result}")
            except Exception as e:
                display_result = f"Error: {equation}"
                print(f"Math error: {e}")

        else:
            display_result = "No drawing found on canvas!"

    # ----------------------------------------------------------
    #  [D] Correction mode – teach the model from its mistakes
    # ----------------------------------------------------------
    elif key == ord('d'):
        if not last_segments:
            print("First press 'a' to analyze, then 'd' to correct.")
        elif last_binary_image is None:
            print("Error: No binary image stored.")
        else:
            print("\n" + "=" * 50)
            print("   CORRECTION MODE")
            print("=" * 50)
            print("  ENTER/SPACE = Confirm (correct)")
            print("  0-9         = Correct digit")
            print("  p=+  m=-  x=*  b=/")
            print("  j = JOIN segments (merge with next)")
            print("  k = SPLIT segment (cut in half)")
            print("  ESC = Cancel")
            print("=" * 50 + "\n")

            # --- Helper: crop a region from binary image and produce 28x28 ---
            def crop_to_28x28(binary_img, bx, by, bw, bh):
                """Crop a region, add padding, invert, and resize to 28x28."""
                region = binary_img[by:by + bh, bx:bx + bw]
                padding = 20
                padded = cv2.copyMakeBorder(
                    region, padding, padding, padding, padding,
                    cv2.BORDER_CONSTANT, value=0
                )
                padded = cv2.bitwise_not(padded)
                return cv2.resize(padded, (28, 28))

            # --- Helper: save a corrected 28x28 image to the dataset ---
            def save_correction(img_28, correct_char):
                """Save the image to the appropriate dataset folder. Returns saved path."""
                folder_name = CHAR_TO_FOLDER.get(correct_char)
                if not folder_name:
                    return None
                target_dir = os.path.join("dataset", folder_name)
                os.makedirs(target_dir, exist_ok=True)
                timestamp = int(time.time() * 1000)
                filename = f"correction_{timestamp}.png"
                full_path = os.path.join(target_dir, filename)
                cv2.imwrite(full_path, img_28)
                return full_path

            # --- Helper: display a segment and wait for a keypress ---
            def show_segment(img_28, label, title, hint=""):
                """Show an enlarged segment with overlay text. Returns the key code."""
                enlarged = cv2.resize(img_28, (280, 280),
                                      interpolation=cv2.INTER_NEAREST)
                colored = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR)
                cv2.putText(colored, f"Prediction: {label}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)
                cv2.putText(colored, title,
                            (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)
                if hint:
                    cv2.putText(colored, hint,
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (200, 200, 200), 1)
                cv2.imshow("CORRECTION MODE", colored)
                return cv2.waitKey(0) & 0xFF

            correction_count = 0
            i = 0
            merge_queue = []

            while i < len(last_segments):
                img_28, pred_label, pred_idx, bbox = last_segments[i]
                title = f"Segment {i + 1}/{len(last_segments)}"
                hint = "ENTER=Ok | j=Join | k=Split | ESC=Exit"

                pressed = show_segment(img_28, pred_label, title, hint)

                # === ESC – cancel ===
                if pressed == 27:
                    print("Correction cancelled.")
                    merge_queue = []
                    break

                # === j – JOIN (merge with next segment) ===
                elif pressed == ord('j'):
                    merge_queue.append((img_28, pred_label, pred_idx, bbox))
                    print(f"  Segment {i + 1}: '{pred_label}' -> added to merge queue "
                          f"[{len(merge_queue)} segments]")
                    i += 1
                    continue

                # === k – SPLIT (cut segment in two) ===
                elif pressed == ord('k'):
                    merge_queue = []
                    bx, by, bw, bh = bbox
                    print(f"  Segment {i + 1}: SPLIT mode – using vertical projection...")

                    region = last_binary_image[by:by + bh, bx:bx + bw]

                    # Vertical projection: count white pixels per column
                    projection = np.sum(region, axis=0) / 255

                    # Find the split point in the middle 20-80% zone
                    start = max(1, int(bw * 0.2))
                    end = min(bw - 1, int(bw * 0.8))
                    if start >= end:
                        start, end = 1, bw - 1

                    mid_zone = projection[start:end]
                    if len(mid_zone) == 0:
                        print("    Cannot split: insufficient width.")
                        i += 1
                        continue

                    split_x = start + np.argmin(mid_zone)
                    print(f"    Split point: x={split_x} (total width: {bw})")

                    # Left half
                    left_img = crop_to_28x28(last_binary_image, bx, by, split_x, bh)
                    left_key = show_segment(left_img, "?",
                                            f"LEFT HALF (Seg {i + 1})",
                                            "Enter the correct character")

                    if left_key == 27:
                        print("    Split cancelled.")
                        break
                    elif left_key in KEY_TO_CHAR:
                        left_char = KEY_TO_CHAR[left_key]
                        path = save_correction(left_img, left_char)
                        if path:
                            correction_count += 1
                            print(f"    Left half: saved as '{left_char}' -> {path}")

                    # Right half
                    right_img = crop_to_28x28(
                        last_binary_image, bx + split_x, by, bw - split_x, bh
                    )
                    right_key = show_segment(right_img, "?",
                                             f"RIGHT HALF (Seg {i + 1})",
                                             "Enter the correct character")

                    if right_key == 27:
                        print("    Split cancelled.")
                        break
                    elif right_key in KEY_TO_CHAR:
                        right_char = KEY_TO_CHAR[right_key]
                        path = save_correction(right_img, right_char)
                        if path:
                            correction_count += 1
                            print(f"    Right half: saved as '{right_char}' -> {path}")

                    i += 1
                    continue

                # === ENTER/SPACE – confirm prediction ===
                elif pressed in (13, 32):
                    if merge_queue:
                        merge_queue.append((img_28, pred_label, pred_idx, bbox))
                        all_boxes = [seg[3] for seg in merge_queue]
                        min_x = min(b[0] for b in all_boxes)
                        min_y = min(b[1] for b in all_boxes)
                        max_x = max(b[0] + b[2] for b in all_boxes)
                        max_y = max(b[1] + b[3] for b in all_boxes)
                        print(f"  {len(merge_queue)} segments merged -> CONFIRMED (no save)")
                        merge_queue = []
                    else:
                        print(f"  Segment {i + 1}: '{pred_label}' -> CONFIRMED")
                    i += 1
                    continue

                # === Character key – correct the prediction ===
                elif pressed in KEY_TO_CHAR:
                    correct_char = KEY_TO_CHAR[pressed]

                    if merge_queue:
                        # Merge all queued segments + current one
                        merge_queue.append((img_28, pred_label, pred_idx, bbox))
                        all_boxes = [seg[3] for seg in merge_queue]
                        min_x = min(b[0] for b in all_boxes)
                        min_y = min(b[1] for b in all_boxes)
                        max_x = max(b[0] + b[2] for b in all_boxes)
                        max_y = max(b[1] + b[3] for b in all_boxes)

                        merged_img = crop_to_28x28(
                            last_binary_image,
                            min_x, min_y, max_x - min_x, max_y - min_y
                        )
                        path = save_correction(merged_img, correct_char)
                        if path:
                            correction_count += 1
                            labels = "+".join([seg[1] for seg in merge_queue])
                            print(f"  MERGED: [{labels}] -> '{correct_char}'")
                            print(f"           Saved: {path}")
                        merge_queue = []
                    else:
                        # Single segment correction
                        if correct_char == pred_label:
                            print(f"  Segment {i + 1}: '{pred_label}' -> Already correct")
                        else:
                            path = save_correction(img_28, correct_char)
                            if path:
                                correction_count += 1
                                print(f"  Segment {i + 1}: '{pred_label}' -> "
                                      f"'{correct_char}' CORRECTED")
                                print(f"           Saved: {path}")
                    i += 1
                    continue
                else:
                    # Invalid key, show the same segment again
                    continue

            cv2.destroyWindow("CORRECTION MODE")

            if correction_count > 0:
                print(f"\n>>> {correction_count} correction(s) saved!")
                print(">>> Press 'r' to retrain the model.\n")
            else:
                print("\nNo corrections were made.\n")

    # ----------------------------------------------------------
    #  [R] Retrain the model with updated dataset
    # ----------------------------------------------------------
    elif key == ord('r'):
        print("\n" + "=" * 50)
        print("   RETRAINING MODEL...")
        print("=" * 50)
        print("This may take a few minutes. Please wait...\n")

        try:
            train_ds = tf.keras.utils.image_dataset_from_directory(
                "dataset",
                validation_split=0.2,
                subset="training",
                seed=123,
                color_mode="grayscale",
                image_size=(28, 28),
                batch_size=32
            )
            val_ds = tf.keras.utils.image_dataset_from_directory(
                "dataset",
                validation_split=0.2,
                subset="validation",
                seed=123,
                color_mode="grayscale",
                image_size=(28, 28),
                batch_size=32
            )

            class_names = train_ds.class_names
            print(f"Class order: {class_names}")

            rescale = tf.keras.layers.Rescaling(1.0 / 255)
            train_ds = train_ds.map(lambda x, y: (rescale(x), y))
            val_ds = val_ds.map(lambda x, y: (rescale(x), y))

            model.fit(train_ds, validation_data=val_ds, epochs=5)

            model.save('calculator_brain.h5')
            print("\n>>> Model retrained and saved successfully!")
            print(">>> It now incorporates your corrections.\n")

        except Exception as e:
            print(f"Training error: {e}")

    # ----------------------------------------------------------
    #  [Q] Quit
    # ----------------------------------------------------------
    elif key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()