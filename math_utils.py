import numpy as np
import cv2

class ViewTransformer:
	def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
		source = source.astype(np.float32)
		target = target.astype(np.float32)
		self.m = cv2.getPerspectiveTransform(source, target)

	def transform_points(self, points: np.ndarray) -> np.ndarray:
		if points.size == 0:
			return points

		reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
		transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
		return transformed_points.reshape(-1, 2)
	
def linear_interpolation(y, y1, x1, y2, x2):
	x = x1 + ((x2 - x1) * (y - y1) / (y2 - y1))
	return x

def count_vertical_lines_in_left_and_right_of_car(car_bottom_center_xy,line_end_coords):
	# line_end_coords: list of lists of [[x1,y1],[x2,y2]] begin end position pairs
	left_line_count = 0
	right_line_count = 0
	estimated_line_coords = []
	car_x, car_y = car_bottom_center_xy
	for line_coord in line_end_coords:
		begin_coord, end_coord = line_coord
		x_begin, y_begin = begin_coord
		x_end, y_end = end_coord
		# estimate the x position of the line at the height level of the car's current position
		line_x_at_y = linear_interpolation(car_y,y_begin,x_begin,y_end,x_end)
		estimated_line_coords.append([line_x_at_y,car_y])
		if line_x_at_y < car_x:
			left_line_count+=1
		else:
			right_line_count+=1
	return left_line_count, right_line_count, estimated_line_coords

def estimate_current_lane_x1_x2(car_bottom_center_xy, estimated_line_coords):
	# estimated_line_coords: lines' xy values at the height level of the car's current position
	car_x = car_bottom_center_xy[0]
	line_x_values = [i[0] for i in estimated_line_coords]
	line_x_values_sorted = sorted(line_x_values)
	for i,i1 in zip(line_x_values_sorted,line_x_values_sorted[1:]):

		# if i<= car_x <= i1:
		# 	return i,i1,car_x
		if i<= car_x <= i1:
			return i,i1,car_x
		elif i1>=i>=car_x: # detected car is a false positive or out of selected road ROI, on the left of leftmost line
			return -1,0,car_x
		else:	 # detected car is a false positive or out of selected road ROI, on the right of rightmost line
			return len(line_x_values),len(line_x_values)+1,car_x
		# TODO: Instead of handling the out of ROI cars this way, a filtering detections inside road ROI can be implemented.
		
def normalize_car_position_in_lane(x1,x2,car_x):
	return (car_x-x1)/(x2-x1)
		

