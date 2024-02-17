import cv2

def plot_lines(img,line_coord_list):
    for coord in line_coord_list:
        cv2.line(img, *coord,  color=(255, 0, 0), thickness=2)
    return img

def plot_points(frame,coords):
    for coord in coords:
        x,y = coord
        x=int(x)
        y=int(y)
        frame = cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    return frame

def draw_label(scene,track,label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale= 1.1
    text_thickness = 2
    top_center_coord = (int((track[0]+track[2])/2),int(track[1]))
    text = (label)
    if label == "up":
        color = (0,0,255)
    elif label == "down":
        color = (255,0,0)
    elif label == "left_lane_change":
        color = (255,0,255)
    else:
        color = (0,255,0)
    cv2.putText(
        img=scene,
        text=text,
        org=top_center_coord,
        fontFace=font,
        fontScale=text_scale,
        color=color,
        thickness=text_thickness,
        lineType=cv2.LINE_AA,
    )
    return scene
