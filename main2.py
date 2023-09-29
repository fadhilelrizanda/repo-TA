import os
import random
import time
import cv2
from ultralytics import YOLO
from collections import deque
import time
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# video_path = "./data_video/camera1_test_2_cam_3.mp4"
# video_out_path = "./predicted_video/camera1_test_2_cam_3.mp4"

filenames_vid = "camera1_test_2_cam_20.mp4"
video_path = "./data_video/part2/camera2_test_2_cam_20.mp4"
video_out_path = "./predicted_video/"+filenames_vid
loc_distance = 0
camera1_stat = False
if camera1_stat :
    loc_distance = 9.46
    #* First coordinate
    count_line_start = (244, 531) 
    count_line_end =(1466, 487)  

    speed_line_start = (221, 848) 
    speed_line_end = (1905, 772) 

    offset_y = 100
    base_y_coordinate = 848 - offset_y
    base_y_coordinate2 = 772 -offset_y

    y_distance = 300

    x_coordinate1 =  521
    x_coordinate2 = 1805
    #* Second Coordinate

    # speed_line_start = (x_coordinate1, base_y_coordinate) 
    # speed_line_end = (x_coordinate2, base_y_coordinate2) 

    # count_line_start = (x_coordinate1, base_y_coordinate-y_distance) 
    # count_line_end =(x_coordinate2, base_y_coordinate2-y_distance)  

    base_line_1_a_x,base_line_1_a_y  = (224,848)
    base_line_2_b_x,base_line_2_b_y  = (1800,772)

    default_distance = 317
    set_distance = 200
    set_y_offset = 100

    base_line_1_a_y = base_line_1_a_y-set_y_offset
    base_line_2_b_y = base_line_2_b_y-set_y_offset


    speed_line_start = (base_line_1_a_x, base_line_1_a_y) # 317
    speed_line_end = (base_line_2_b_x, base_line_2_b_y) 

    count_line_start = (base_line_1_a_x, base_line_1_a_y -set_distance)  # 317
    count_line_end =(base_line_2_b_x, base_line_2_b_y -set_distance)   #285


else:

    loc_distance = 9.7
    x_offset = 200
    y_offset = 350

    base_x, base_y = (1691, 956-y_offset)
    base_x2, base_y2 = (102, 742-y_offset)

    #check slope
    slope = (base_y2-base_y)/(base_x2-base_x)
    print(f"slope : {slope}")

    
    base2_x, base2_y = (1691, 460-y_offset)
    base2_x2 = 958
    base2_y2 = slope*(base2_x2-base2_x)+base2_y 



# calculate distance (5m)
    y_line_pred = slope*(base_x-base2_x)+base2_y
    distance = abs(base_y-y_line_pred)
    print(distance)

    #reducing 
    red_base2_x, red_base2_y = (1791, 460+200 -y_offset)
    red_base2_x2 = 458
    red_base2_y2 = slope*(red_base2_x2-red_base2_x)+red_base2_y 


    y_line_pred_red = slope*(base_x-red_base2_x)+red_base2_y
    distance_red = abs(base_y-y_line_pred_red)
    print(distance_red)
    speed_line_start = (base_x,base_y)
    speed_line_end = (base_x2,base_y2)

    # count_line_start = (base2_x, int(base2_y)) 
    # count_line_end =(base2_x2, int(base2_y2)) 

#red
    count_line_start = (red_base2_x, int(red_base2_y)) 
    count_line_end =(red_base2_x2, int(red_base2_y2)) 




cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model = YOLO("./model_data/best_v3.pt")

cfg_class = [""]

max_cosine_distance = 0.4
nn_budget = None
nms_max_overlap = 1.0
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
tracker = Tracker(metric)
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

colors = [(random.randint(0, 255), random.randint(0, 255),
           random.randint(0, 255)) for j in range(10)]

kendaraan_besar_masuk = 0
kendaraan_besar_keluar = 0
kendaraan_kecil_masuk = 0
kendaran_kecil_keluar = 0

detection_threshold = 0.5

memory = {}
already_counted = deque(maxlen=50)
num_det_up = 0
num_det_down = 0
counted_state = False
frame_size = (640,640)

def orientation(pointA, pointB, pointC):
    return (pointB[0]-pointA[0])*(pointC[1]-pointA[1]) - (pointB[1]-pointA[1])*(pointC[0]-pointA[0])


def do_line_segments_intersect(pointA, pointB, pointC, pointD):
    orientation1 = orientation(pointA, pointB, pointC)
    orientation2 = orientation(pointA, pointB, pointD)
    orientation3 = orientation(pointC, pointD, pointA)
    orientation4 = orientation(pointC, pointD, pointB)

    if (orientation1 * orientation2 < 0) and (orientation3 * orientation4 < 0):
        return True
    return False


def grad_eq(grad, x, x1, y1):
    y = grad*(x-x1)+y1
    return y

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def point_locr(linePointA, linePointB, point):
    grad = (linePointB[1]-linePointA[1])/(linePointB[0]-linePointA[0])
    y_point = grad_eq(grad, point[0], linePointA[0], linePointA[1])

    if y_point < point[1]:
        condition = True
    else:
        condition = False

    return condition

class_file = "./kendaraan.names"

speed_state = False

start_speed_dict ={}
end_speed_dict = {}
frame_iter = 0
last_speed = 0
while ret:
    frame_iter +=1
    print(frame_iter)
    ret, frame = cap.read()

    # preprocessing
    frame_height, frame_width,_ = frame.shape
    # frame = cv2.resize(frame,(640,640))
    results = model(frame)
    class_names = read_class_names(class_file)

    for result in results:
        # print(result.boxes.data)
        # print(type(result))
        detections_obj = []
        class_obj = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections_obj.append([x1, y1, x2, y2, score])
                class_obj.append(class_id)
            else:
                continue
        
        # print(detections_obj)
        # print(len(detections_obj))
        if len(detections_obj) == 0:
            continue
        bboxes = np.asarray([d[:-1] for d in detections_obj])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections_obj]
        features = encoder(frame, bboxes)
        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], class_obj[bbox_id], features[bbox_id]))

        tracker.predict()
        tracker.update(dets)

        # tracker.update(frame, detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            # conf_score = track.confidence
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            if class_names[class_name] == "truck":
                use_colors = (0,0,150)
            else:
                use_colors = (0,150,0)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(
                x2), int(y2)), use_colors, 3)
            midpoint_x = int(abs((x1+x2)/2))
            midpoint_y = int(abs((y1+y2)/2))
            current_midpoint = (midpoint_x, midpoint_y)

            if track.track_id not in memory:
                memory[track.track_id] = deque(maxlen=2)

            memory[track.track_id].append(current_midpoint)
            previous_midpoint = memory[track.track_id][0]

            cv2.line(frame, current_midpoint,
                     previous_midpoint, (0, 255, 0), 2)

            # put text in bounding boxes

            text = f'c:{class_names[class_name]}'
            font = cv2.FONT_HERSHEY_SIMPLEX  # Choosing a font face
            font_scale = 0.4
            font_color = (255, 255, 255)  # Green color
            font_thickness = 1
            line_type = cv2.LINE_AA

            cv2.rectangle(frame, (int(x1)-3, int(y1)), (int(x2)+3, int(y1)-10), (use_colors), -1)
            cv2.putText(frame, text, (int(x1), int(y1)), font,
                        font_scale, font_color, font_thickness, line_type)
            #count object
            if do_line_segments_intersect(count_line_start, count_line_end, current_midpoint, previous_midpoint):
                # print("Line segments intersect.")
                counted_state = True
                start_speed_dict[track_id] = frame_iter

                if point_locr(count_line_start, count_line_end, previous_midpoint):
                    num_det_up += 1
                    if class_names[class_name] == "truck":
                        kendaraan_besar_keluar += 1
                    else :
                        kendaran_kecil_keluar += 1
                else:
                    num_det_down += 1
                    if class_names[class_name] == "truck":
                        kendaraan_besar_masuk += 1
                    else :
                        kendaraan_kecil_masuk += 1

            if do_line_segments_intersect(speed_line_start, speed_line_end, current_midpoint, previous_midpoint):
                # print("Line segments intersect.")
                if track_id in start_speed_dict :
                    if point_locr(speed_line_start, speed_line_end, previous_midpoint) :
                        print("passed")
                    else:
                        speed_state = True
                        end_speed_dict[track_id] = frame_iter
                        try:
                            last_speed = (loc_distance/((end_speed_dict[track_id]-start_speed_dict[track_id])/25))*3.6
                        except ZeroDivisionError as e:
                            last_speed = 0
                        except Exception as e:
                            # Handle other exceptions here (if needed)
                            print(f"An error occurred: {e}")
            

    # Draw Square in right corner
    print(frame_width)
    print(frame_height)
    top_left = (frame_width - 600, 20)
    bottom_right = (frame_width - 20, 200)

    # Draw a filled rectangle
    cv2.rectangle(frame, top_left, bottom_right, (150, 0, 0), -1)

    print(f'k_masuk : {num_det_down} k_keluar : {num_det_up} ')
    # Assuming you have defined the values for num_det, x1, and y1
    print(start_speed_dict)
    print(end_speed_dict)
    print(f'last speed : {last_speed}')
    text = f'k_masuk : {num_det_down} k_keluar : {num_det_up} '

    font = cv2.FONT_HERSHEY_SIMPLEX  # Choosing a font face
    font_scale = 1
    font_color = (0, 255, 0)  # Green color
    font_thickness = 2
    line_type = cv2.LINE_AA

    cv2.putText(frame, text, (frame_width - 550, int(80)), font,
                font_scale, font_color, font_thickness, line_type)
    
    text = f'K_B_masuk : {kendaraan_besar_masuk}  K_B_keluar : {kendaraan_besar_keluar}'
    font = cv2.FONT_HERSHEY_SIMPLEX  # Choosing a font face
    font_scale = 1
    font_color = (0, 255, 0)  # Green color
    font_thickness = 2
    line_type = cv2.LINE_AA

    cv2.putText(frame, text, (frame_width - 550, int(110)), font,
                font_scale, font_color, font_thickness, line_type)
    
    text = f'K_K_masuk : {kendaraan_kecil_masuk}  K_K_keluar : {kendaran_kecil_keluar}'
    font = cv2.FONT_HERSHEY_SIMPLEX  # Choosing a font face
    font_scale = 1
    font_color = (0, 255, 0)  # Green color
    font_thickness = 2
    line_type = cv2.LINE_AA

    cv2.putText(frame, text, (frame_width - 550, int(140)), font,
                font_scale, font_color, font_thickness, line_type)
    



    text = f'Last Speed : {last_speed}'

    font = cv2.FONT_HERSHEY_SIMPLEX  # Choosing a font face
    font_scale = 1
    font_color = (0, 255, 0)  # Green color
    font_thickness = 2
    line_type = cv2.LINE_AA

    cv2.putText(frame, text, (frame_width - 550, int(180)), font,
                font_scale, font_color, font_thickness, line_type)

    if counted_state:
        cv2.line(frame, count_line_start, count_line_end, (0, 0, 255), 2)
    else:
        cv2.line(frame, count_line_start, count_line_end, (0, 255, 0), 2)

    if speed_state:
        cv2.line(frame, speed_line_start, speed_line_end, (0, 0, 255), 2)
    else:
        cv2.line(frame, speed_line_start, speed_line_end, (0, 255, 0), 2)
    speed_state = False

    counted_state = False
    im_show = cv2.resize(frame, (960, 540))
    cv2.imshow("frame", im_show)
    key = cv2.waitKey(1)  # Wait for a key event for 1 millisecond

    if key == 27:  # Exit when the 'Esc' key is pressed
        break

    cap_out.write(frame)


cap.release()
cap_out.release()
cv2.destroyAllWindows()
