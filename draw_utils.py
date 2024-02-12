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
