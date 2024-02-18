
from collections import deque, Counter


from math_utils import ViewTransformer, linear_interpolation, count_vertical_lines_in_left_and_right_of_car,estimate_current_lane_x1_x2 \
                     , normalize_car_position_in_lane


class CarTrack:
    def __init__(self, track_id, init_coord) -> None:
        self.track_id = track_id
        self.init_coord = init_coord
        self.xy_bottom_center_history = deque(maxlen=100)
        self.vertical_direction_history = deque(maxlen=100)
        self.horizontal_direction_history = deque(maxlen=100)
        self.lane_number_history = deque(maxlen=100)
        self.lane_normalized_position_history = deque(maxlen=100)

        self.first_bottom_center_coord = self.get_bottom_center_coord(init_coord)
        self.xy_bottom_center_history.append(self.first_bottom_center_coord)
    
    def add_coordinate(self, coord, line_end_coords):
        """
        This method does not determine the final status decision of lane changes
        or vertical direction, instead, it calculates these values for last frame
        and stores them in history. final status is calculated using past status information 
        """
        bottom_center_coord = self.get_bottom_center_coord(coord)
        self.xy_bottom_center_history.append(bottom_center_coord)

        left, right, estimated_line_coords = count_vertical_lines_in_left_and_right_of_car(bottom_center_coord,line_end_coords)
        current_lane_no = left  # If there is one line staying in leftside of car, car is in 1st lane 
        # Add the current lane number to the lane_number_history based on the left and right line counts
        self.lane_number_history.append(current_lane_no)
        x1,x2,car_x = estimate_current_lane_x1_x2(bottom_center_coord,estimated_line_coords)
        car_norm_coordinate_in_lane = normalize_car_position_in_lane(x1,x2,car_x)
        # Add car's normalized position in its current lane to the history
        """
        below operation is done for making horizontal position change ratio calculation easier,
        for example, if there is 1 line in leftside of car, the car is in leftmost lane (lets say 1st lane), 
        and the car is at the center of the lane; then its normalized position is 0.5 unit distance from
        leftmost line in scene. Another example, if the car is on 3rd lane (from left) and at 0.7 unit distant
        from 3rd lane left borderline, then its position is 2 entire lane and 0.7 unit lane; 2.7 unit. 
        
        So the horizontal normalized position formula is :  (current_lane_no - 1) + car_norm_coordinate_for_current_lane

        """
        car_norm_coordinate_in_lane = (current_lane_no - 1) + car_norm_coordinate_in_lane
        self.lane_normalized_position_history.append(car_norm_coordinate_in_lane)

        
        if len(self.xy_bottom_center_history) >= 2:
            
            # Initial operations for vertical direction calculation
            # Calculate the difference in y coordinate between the last two positions
            y_difference = self.xy_bottom_center_history[-1][1] - self.xy_bottom_center_history[-2][1]
            # Add the direction change to the vertical_direction_history based on the y difference
            # In below operations, we append a +1 value for every x or y increase and 
            # append a -1 value for every x or y decrease.
            # x increase means rightward movement and, y increase means downward direction
            if y_difference < 0:
                self.vertical_direction_history.append(-1)  # Going upward
            elif y_difference > 0:
                self.vertical_direction_history.append(1) # Going downward
            else:
                self.vertical_direction_history.append(0)  # Not moving vertically
            # self.vertical_direction_history.append(-y_difference)
                
        if len(self.lane_normalized_position_history) >= 2:
            # Initial operations for lane change calculation
            # Estimate the current lane state of the car (which lane the car is moving in)
            lane_normalized_position_change = self.lane_normalized_position_history[-1] - self.lane_normalized_position_history[-2]
            if lane_normalized_position_change < 0:
                self.horizontal_direction_history.append(-1) # Going leftward
            elif lane_normalized_position_change > 0:
                self.horizontal_direction_history.append(1) # Going rightward
            else: 
                self.horizontal_direction_history.append(0) # No horizontal position change

            
    def get_bottom_center_coord(self,coord):
        return (int((coord[0]+coord[2])/2),int(coord[3]))
    def check_direction(self,top_line_y,bottom_line_y):
        if len(self.xy_bottom_center_history) < 2:
            return ""
        
        # Check the last 100 direction changes
        last_direction_changes = list(self.vertical_direction_history)
        # Calculate the total direction change
        total_change = sum(last_direction_changes)
        last_y_coordinate = self.xy_bottom_center_history[-1][1]
        is_between_horizontal_lines = (bottom_line_y >=last_y_coordinate >= top_line_y)
        if is_between_horizontal_lines:
            a = 1
        if total_change < 0 and is_between_horizontal_lines:
            return "up"
        elif total_change > 0 and is_between_horizontal_lines:
            return "down"
        else:
            return ""
    def check_lane_change(self):
        if len(self.xy_bottom_center_history) < 2:
            return ""
        
        # Check the last 100 direction changes
        last_horizontal_directions_left_right = list(self.horizontal_direction_history)
        last_100_lane_numbers = list(self.lane_number_history)
        last_10_lane_numbers = list(self.lane_number_history)[-10:]
        last_100_most_common_lane = Counter(last_100_lane_numbers).most_common(1)[0][0]
        last_10_most_common_lane = Counter(last_10_lane_numbers).most_common(1)[0][0]
        
        # Calculate the total direction change for last 100 horizontal direction change
        total_change = sum(last_horizontal_directions_left_right)
        if (total_change < 0) and (last_100_most_common_lane > last_10_most_common_lane):
            return "left_lane_change"
        elif (total_change > 0) and (last_100_most_common_lane < last_10_most_common_lane):
            return "right_lane_change"
        else:
            return ""