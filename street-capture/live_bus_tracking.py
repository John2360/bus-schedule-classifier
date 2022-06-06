"""
Use webcam and bus model to live flag buses
"""
import os
import sys
import cv2
import numpy as np
from datetime import datetime

p = os.path.abspath('.')
sys.path.insert(1, p)
from sort.sort import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file('classifier/model/pipeline.config')
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('classifier/model/checkpoint', 'ckpt-0')).expect_partial()
print("Successfully loaded model")

#create instance of SORT
mot_tracker = Sort()
bus_tracker = {}
print("Started SORT tracking")

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])


label_map_path = 'classifier/model/label_map.pbtxt'
category_index=label_map_util.create_category_index_from_labelmap(label_map_path ,use_display_name=True)

cap = cv2.VideoCapture('test.mp4')
print("Opened camera")

while True:
    # Read frame from camera
    ret, image_np = cap.read()

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)


    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    _, labels = viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.3,
          agnostic_mode=False)
    
    labels = np.array(labels)
    if labels.size == 0:
        labels = np.empty((0, 5))
    track_bbs_ids = mot_tracker.update(labels)

    for bus in track_bbs_ids:
        if bus[4] not in bus_tracker:
            bus_tracker[bus[4]] = [(bus[0], bus[2])]
        else:
            bus_tracker[bus[4]].append((bus[0], bus[2]))

    buses_movement = []
    buses_not_detected = set(bus_tracker) - set(x[4] for x in track_bbs_ids)

    for bus in buses_not_detected:
        if len(bus_tracker[bus]) >= 2:
            bus_x_positions = bus_tracker[bus]
            bus_direction_tracker = []

            while len(bus_x_positions) != 0:
                old_x1, old_x2 = bus_x_positions.pop()

                if len(bus_x_positions) != 0:
                    new_x1, new_x2 = bus_x_positions.pop()

                    if new_x1-old_x1>0 and new_x2-old_x2>0:
                        bus_direction_tracker.append(1)
                    elif new_x1-old_x1<0 and new_x2-old_x2<0:
                        bus_direction_tracker.append(-1)
                    else:
                        bus_direction_tracker.append(0)
            
            average = sum(bus_direction_tracker)/len(bus_direction_tracker)

            if average > 0:
                buses_movement.append("up")
            elif average < 0:
                buses_movement.append("down")
        
        # remove from tracker as is old
        del bus_tracker[bus]

    if len(buses_movement) != 0:
        cv2.imwrite("./classifier/review/images/FrontStreet_"+str(datetime.timestamp(datetime.now()))+".jpg", cv2.resize(image_np_with_detections, (800, 600)))

    # Display output
    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

    c = cv2.waitKey(1) % 256
    if c == ord('q'):
        print("Goodbye!")
        break  # esc to quit

cap.release()
cv2.destroyAllWindows()