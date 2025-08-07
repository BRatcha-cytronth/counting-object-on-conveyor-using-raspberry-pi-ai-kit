from picamera2 import Picamera2
import cv2
import numpy as np
from hailo_platform.pyhailort.pyhailort import (
    VDevice, HEF,
    InputVStreams, OutputVStreams,
    InputVStreamParams, OutputVStreamParams
)

# --- Config ---
hef_path = "/home/pi/path/to/your/model.hef"
conf_threshold = 0.8
class_name = "your_model_class_name"
count_line_x = 320
count_radius = 5           # รัศมีรอบเส้นสำหรับการนับ
max_track_distance = 100   # Euclidean distance threshold
frame_width, frame_height = 640, 480
min_box_width = 100        # px

# --- State ---
counted_ids = set()
object_counter = 0
next_object_id = 0
object_tracks = {}

# --- Postprocessing Function ---
def yolo_hailo_postprocess(raw_output, conf_threshold=0.3):
    raw_output = np.array(raw_output).ravel()
    detections = []
    for i in range(0, len(raw_output), 5):
        if i + 4 >= len(raw_output):
            break
        y_min, x_min, y_max, x_max, conf = raw_output[i:i + 5]
        if conf < conf_threshold:
            continue
        # scale to pixel
        x_min = int(x_min * frame_width)
        x_max = int(x_max * frame_width)
        y_min = int(y_min * frame_height)
        y_max = int(y_max * frame_height)

        # filter Object ที่เล็กเกินไป
        if x_max - x_min < min_box_width:
            continue

        detections.append((conf, (x_min, y_min, x_max, y_max)))
    return detections

# --- Object Tracker ---
def track_objects(detections):
    global next_object_id, object_counter
    new_tracks = {}

    for conf, (xmin, ymin, xmax, ymax) in detections:
        cx = int((xmin + xmax) / 2)
        cy = int((ymin + ymax) / 2)

        matched_id = None
        for object_id, (prev_cx, prev_cy) in object_tracks.items():
            dist2 = (cx - prev_cx)**2 + (cy - prev_cy)**2
            if dist2 < max_track_distance**2:
                matched_id = object_id
                break

        # new object
        if matched_id is None:
            # ถ้าเกิดหลังเส้น + รัศมี → ข้ามไม่ต้องนับ
            if cx > count_line_x + count_radius:
                continue
            matched_id = next_object_id
            next_object_id += 1

        new_tracks[matched_id] = (cx, cy)

        # count ถ้ายังไม่เคยนับ และจุดกลางอยู่ในรัศมี ±5 px ของเส้น
        if matched_id not in counted_ids and abs(cx - count_line_x) <= count_radius:
            counted_ids.add(matched_id)
            object_counter += 1

    return new_tracks

# --- Main Application ---
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (frame_width, frame_height)}))
picam2.start()
picam2.set_controls({"AfMode": 1})

hef = HEF(hef_path)
with VDevice() as device:
    network_group = device.configure(hef)[0]
    input_info = hef.get_input_vstream_infos()[0]
    output_info = hef.get_output_vstream_infos()[0]
    input_shape = tuple(input_info.shape)  # (H, W, C)

    input_params = InputVStreamParams.make(network_group)
    output_params = OutputVStreamParams.make(network_group)

    with network_group.activate(network_group.create_params()):
        with InputVStreams(network_group, input_params) as input_streams, \
             OutputVStreams(network_group, output_params) as output_streams:

            input_stream = next(iter(input_streams))
            output_stream = next(iter(output_streams))

            print("🚀 Real-time object detection and counting (±5px radius)...")

            while True:
                frame = picam2.capture_array()

                # --- เตรียม input ให้ Hailo ---
                rgb_resized = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_resized = cv2.resize(rgb_resized, (input_shape[1], input_shape[0]))
                input_data = np.ascontiguousarray(rgb_resized).ravel()[np.newaxis, :]

                # --- ส่งเข้า Hailo ---
                input_stream.send(input_data)
                output_data = output_stream.recv()
                output_array = np.copy(output_data[0])

                # --- Postprocess + Tracking ---
                detections = yolo_hailo_postprocess(output_array, conf_threshold)
                object_tracks = track_objects(detections)

                # --- Draw ---
                for object_id, (cx, cy) in object_tracks.items():
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"ID:{object_id}", (cx, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                for conf, (xmin, ymin, xmax, ymax) in detections:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                cv2.line(frame, (count_line_x, 0), (count_line_x, frame.shape[0]), (255, 0, 0), 2)
                cv2.putText(frame, f"Count: {object_counter}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow("Hailo YOLO Real-time", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

cv2.destroyAllWindows()
picam2.stop()
