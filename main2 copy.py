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

video_path = "./data_video/camera1_test_2_cam_3.mp4"
video_path2 = "./data_video/camera2_test_2_cam_3.mp4"
video_out_path = "./predicted_video/camera_combine_test3.mp4"
# video_out_path2 = "./predicted_video/camera2_test_2_cam_3.mp4"



count_line_start = (444, 531)
count_line_end =(1466, 487)

speed_line_start = (621, 848)
speed_line_end = (1905, 772)




cap = cv2.VideoCapture(video_path)
cap2 = cv2.VideoCapture(video_path2)

ret, frame = cap.read()
ret2, frame2 = cap2.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (1366, 768))
# cap_out2 = cv2.VideoWriter(video_out_path2, cv2.VideoWriter_fourcc(*'MP4V'), cap2.get(cv2.CAP_PROP_FPS),
#                           (frame2.shape[1], frame2.shape[0]))

model = YOLO("./model_data/best_v2.pt")

cfg_class = [""]

max_cosine_distance = 0.4
nn_budget = None
nms_max_overlap = 1.0
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
metric2 = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
tracker = Tracker(metric)
tracker2 = Tracker(metric2)

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

colors = [(random.randint(0, 255), random.randint(0, 255),
           random.randint(0, 255)) for j in range(10)]

kendaraan_besar_masuk = 0
kendaraan_besar_keluar = 0
kendaraan_kecil_masuk = 0
kendaraan_kecil_keluar = 0

kendaraan_besar_masuk2 = 0
kendaraan_besar_keluar2 = 0
kendaraan_kecil_masuk2 = 0
kendaraan_kecil_keluar2 = 0

detection_threshold = 0.5

memory = {}
memory2 = {}
already_counted = deque(maxlen=50)
num_det_up = 0
num_det_down = 0
num_det_up2 = 0
num_det_down2 = 0


counted_state = False
counted_state2 = False

cam_1_kb = False
cam_2_kb = False

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

def check_condition(vehicle_cam1, vehicle_cam2):
    if vehicle_cam1> vehicle_cam2:
        return True
    else:
        return False

class_file = "./kendaraan.names"

speed_state = False
speed_state2 = False

start_speed_dict ={}
end_speed_dict = {}
frame_iter = 0
last_speed = 0
last_speed2 = 0
while ret and ret2:


    frame_iter +=1
    print(frame_iter)
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()

    # preprocessing
    frame_height, frame_width,_ = frame.shape
    frame_height2, frame_width2,_ = frame2.shape
    # frame = cv2.resize(frame,(640,640))
    results = model(frame)
    results2 = model(frame2)
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
                        kendaraan_kecil_keluar += 1
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
                        last_speed = (15/((end_speed_dict[track_id]-start_speed_dict[track_id])/25))*3.6

    for result2 in results2:
        # print(result.boxes.data)
        # print(type(result))
        detections_obj = []
        class_obj = []
        for r in result2.boxes.data.tolist():
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

        tracker2.predict()
        tracker2.update(dets)

        # tracker.update(frame, detections)

        for track in tracker2.tracks:
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

            cv2.rectangle(frame2, (int(x1), int(y1)), (int(
                x2), int(y2)), use_colors, 3)
            midpoint_x = int(abs((x1+x2)/2))
            midpoint_y = int(abs((y1+y2)/2))
            current_midpoint = (midpoint_x, midpoint_y)

            if track.track_id not in memory2:
                memory2[track.track_id] = deque(maxlen=2)

            memory2[track.track_id].append(current_midpoint)
            previous_midpoint = memory2[track.track_id][0]

            cv2.line(frame2, current_midpoint,
                     previous_midpoint, (0, 255, 0), 2)

            # put text in bounding boxes

            text = f'c:{class_names[class_name]}'
            font = cv2.FONT_HERSHEY_SIMPLEX  # Choosing a font face
            font_scale = 0.4
            font_color = (255, 255, 255)  # Green color
            font_thickness = 1
            line_type = cv2.LINE_AA

            cv2.rectangle(frame2, (int(x1)-3, int(y1)), (int(x2)+3, int(y1)-10), (use_colors), -1)
            cv2.putText(frame2, text, (int(x1), int(y1)), font,
                        font_scale, font_color, font_thickness, line_type)
            #count object
            if do_line_segments_intersect(count_line_start, count_line_end, current_midpoint, previous_midpoint):
                # print("Line segments intersect.")
                counted_state2 = True
                start_speed_dict[track_id] = frame_iter

                if point_locr(count_line_start, count_line_end, previous_midpoint):
                    num_det_up2 += 1
                    if class_names[class_name] == "truck":
                        kendaraan_besar_keluar2 += 1
                    else :
                        kendaraan_kecil_keluar2 += 1
                else:
                    num_det_down2 += 1
                    if class_names[class_name] == "truck":
                        kendaraan_besar_masuk2 += 1
                    else :
                        kendaraan_kecil_masuk2 += 1

            if do_line_segments_intersect(speed_line_start, speed_line_end, current_midpoint, previous_midpoint):
                # print("Line segments intersect.")
                if track_id in start_speed_dict :
                    if point_locr(speed_line_start, speed_line_end, previous_midpoint) :
                        print("passed")
                    else:
                        speed_state2 = True
                        end_speed_dict[track_id] = frame_iter
                        last_speed2 = (15/((end_speed_dict[track_id]-start_speed_dict[track_id])/25))*3.6
        

    #** Draw Information
    print(frame_width)
    print(frame_height)
    top_left = (frame_width - 600, 20)
    bottom_right = (frame_width - 20, 250)
    top_left2 = (20, 20)
    bottom_right2 = (580, 250)

    # Draw a filled rectangle
    cv2.rectangle(frame, top_left, bottom_right, (150, 0, 0), -1)
    cv2.rectangle(frame2, top_left2, bottom_right2, (150, 0, 0), -1)

    print(f'k_keluar : {num_det_up}  k_masuk : {num_det_down}')
    # Assuming you have defined the values for num_det, x1, and y1
    print(start_speed_dict)
    print(end_speed_dict)
    print(f'last speed : {last_speed}')
    text = f'k_keluar : {num_det_up}  k_masuk : {num_det_down}'

    font = cv2.FONT_HERSHEY_SIMPLEX  # Choosing a font face
    font_scale = 1
    font_color = (0, 255, 0)  # Green color
    font_thickness = 2
    line_type = cv2.LINE_AA

    cv2.putText(frame, text, (frame_width - 550, int(80)), font,
                font_scale, font_color, font_thickness, line_type)
    

    text = f'k_keluar : {num_det_up2}  k_masuk : {num_det_down2}'

    font = cv2.FONT_HERSHEY_SIMPLEX  # Choosing a font face
    font_scale = 1
    font_color = (0, 255, 0)  # Green color
    font_thickness = 2
    line_type = cv2.LINE_AA

    cv2.putText(frame2, text, (70, int(80)), font,
                font_scale, font_color, font_thickness, line_type)
    
    text = f'K_B_masuk : {kendaraan_besar_masuk}  K_B_keluar : {kendaraan_besar_keluar}'
    font = cv2.FONT_HERSHEY_SIMPLEX  # Choosing a font face
    font_scale = 1
    font_color = (0, 255, 0)  # Green color
    font_thickness = 2
    line_type = cv2.LINE_AA

    cv2.putText(frame, text, (frame_width - 550, int(110)), font,
                font_scale, font_color, font_thickness, line_type)
    
    text = f'K_B_masuk : {kendaraan_besar_masuk2}  K_B_keluar : {kendaraan_besar_keluar2}'
    font = cv2.FONT_HERSHEY_SIMPLEX  # Choosing a font face
    font_scale = 1
    font_color = (0, 255, 0)  # Green color
    font_thickness = 2
    line_type = cv2.LINE_AA

    cv2.putText(frame2, text, (70, int(110)), font,
                font_scale, font_color, font_thickness, line_type)
    
    text = f'K_K_masuk : {kendaraan_kecil_masuk}  K_K_keluar : {kendaraan_kecil_keluar}'
    font = cv2.FONT_HERSHEY_SIMPLEX  # Choosing a font face
    font_scale = 1
    font_color = (0, 255, 0)  # Green color
    font_thickness = 2
    line_type = cv2.LINE_AA

    cv2.putText(frame, text, (frame_width - 550, int(140)), font,
                font_scale, font_color, font_thickness, line_type)
    

    
    text = f'K_K_masuk : {kendaraan_kecil_masuk2}  K_K_keluar : {kendaraan_kecil_keluar2}'
    font = cv2.FONT_HERSHEY_SIMPLEX  # Choosing a font face
    font_scale = 1
    font_color = (0, 255, 0)  # Green color
    font_thickness = 2
    line_type = cv2.LINE_AA

    cv2.putText(frame2, text, (70, int(140)), font,
                font_scale, font_color, font_thickness, line_type)
    
    text = f'Last Speed : {last_speed}'
    font = cv2.FONT_HERSHEY_SIMPLEX  # Choosing a font face
    font_scale = 1
    font_color = (0, 255, 0)  # Green color
    font_thickness = 2
    line_type = cv2.LINE_AA

    cv2.putText(frame, text, (frame_width - 550, int(180)), font,
                font_scale, font_color, font_thickness, line_type)


    text = f'Last Speed : {last_speed2}'
    font = cv2.FONT_HERSHEY_SIMPLEX  # Choosing a font face
    font_scale = 1
    font_color = (0, 255, 0)  # Green color
    font_thickness = 2
    line_type = cv2.LINE_AA


    cv2.putText(frame2, text, (70, int(180)), font,
                font_scale, font_color, font_thickness, line_type)


    if kendaraan_besar_keluar2>kendaraan_besar_masuk:
        kendaraan_besar_keluar2-=1

    if kendaraan_besar_keluar>kendaraan_besar_masuk2:
        kendaraan_besar_keluar-=1
    
    if  kendaraan_kecil_keluar>kendaraan_kecil_masuk2:
        kendaraan_kecil_keluar-=1

    if kendaraan_kecil_keluar2>kendaraan_kecil_masuk:
        kendaraan_kecil_keluar2-=1

    camera1_stat_kb = check_condition(kendaraan_besar_masuk2,kendaraan_besar_keluar)
    camera1_stat_kc = check_condition(kendaraan_kecil_masuk2,kendaraan_kecil_keluar)

    camera2_stat_kb = check_condition(kendaraan_besar_masuk,kendaraan_besar_keluar2)
    camera2_stat_kc = check_condition(kendaraan_kecil_masuk,kendaraan_kecil_keluar2)

  


    if camera2_stat_kb:
        txt_stat = "Berhenti!"
        txt_color = (0,0,255)
    elif camera2_stat_kc :
        txt_stat = "Hati-hati!"
        txt_color = (0,255,255)
    else:
        txt_stat = "Tidak Ada kendaraan"
        txt_color = (0,255,0)
    text = f'Status : {txt_stat}'
    font = cv2.FONT_HERSHEY_SIMPLEX  # Choosing a font face
    font_scale = 1
    font_color = txt_color # Green color
    font_thickness = 2
    line_type = cv2.LINE_AA


    cv2.putText(frame2, text, (70, int(220)), font,
                font_scale, font_color, font_thickness, line_type)

    
    if camera1_stat_kb:
        txt_stat = "Berhenti!"
        txt_color = (0,0,255)
    elif camera1_stat_kc :
        txt_stat = "Hati-hati!"
        txt_color = (0,255,255)
    else:
        txt_stat = "Tidak Ada kendaraan"
        txt_color = (0,255,0)
    
    text = f'Status : {txt_stat}'
    font = cv2.FONT_HERSHEY_SIMPLEX  # Choosing a font face
    font_scale = 1
    font_color = txt_color # Green color
    font_thickness = 2
    line_type = cv2.LINE_AA

    cv2.putText(frame, text, (frame_width - 550, int(220)), font,
                font_scale, font_color, font_thickness, line_type)

    
    

    ##**

    ##** accesories line



    #** Draw Line
    if counted_state:
        cv2.line(frame, count_line_start, count_line_end, (0, 0, 255), 2)
    else:
        cv2.line(frame, count_line_start, count_line_end, (0, 255, 0), 2)

    if counted_state2:
        cv2.line(frame2, count_line_start, count_line_end, (0, 0, 255), 2)
    else:
        cv2.line(frame2, count_line_start, count_line_end, (0, 255, 0), 2)

    if speed_state:
        cv2.line(frame, speed_line_start, speed_line_end, (0, 0, 255), 2)
    else:
        cv2.line(frame, speed_line_start, speed_line_end, (0, 255, 0), 2)
    speed_state = False

    if speed_state2:
        cv2.line(frame2, speed_line_start, speed_line_end, (0, 0, 255), 2)
    else:
        cv2.line(frame2, speed_line_start, speed_line_end, (0, 255, 0), 2)
    speed_state2 = False

    #**
    counted_state = False
    counted_state2 = False
    im_show = cv2.resize(frame, (683, 768))
    im_show2 = cv2.resize(frame2, (683, 768))
    merged_frame = np.hstack((im_show, im_show2))
    cv2.imshow("frame", merged_frame)
    # cv2.imshow("frame2", im_show2)
    key = cv2.waitKey(1)  # Wait for a key event for 1 millisecond

    if key == 27:  # Exit when the 'Esc' key is pressed
        break

    cap_out.write(merged_frame)


cap.release()
cap_out.release()
cv2.destroyAllWindows()
