# Car Direction Counting

**Car direction and lane change estimation on non-static camera**
## 📄 Algorithm Workflow

| |Main Steps of the Pipeline     |
|-|:---------------:|
|1|Generate a video reader and writer instance  |
|2|Initialize OCSort Tracker |
|3|Set Optical Flow parameters to stabilize the lane borders in the the video   |
|4|Select lane borders and reference stabilization points or load from config   |
|5|Load pretrained YOLOv5s6 object detection model |
|6|Start the detection and tracking inference loop |


| |Detection and Tracking Inference Loop     |
|-|:---------------:|
|1|In each iteration, grab a frame and update the timer  |
|2|Calculate the corrected lane coordinates using optical flow of reference points|
|3|Shift the previous lane coordines to the corrected coordinates using ViewTransformer |
|4| Run object detector model|
|5| Feed OCSort tracker with the detector results and get track ids for detected bboxes|
|6| Initialize CarTrack class instance which keeps the prev coordinates if not initialized before|
|7| Add the current track coordinate to the CarTrack history, this method also adds vertical and horizontal shift information for last 2 frame
|8| Estimate the vertical and horizontal movement using CarTrack object's .check_direction() and .check_lane_change() methods, this method leverages the historical movement information |
|9| Calculate the elapsed time with respect to assumed beginning time and add the car state to the result dictionary|



## Dependencies
> [!TIP]
> _It's recommended to use this script with conda or venv virtual environments_



```bash
python3 -m pip install -r requirements.txt
```

## Usage
> [!IMPORTANT]
> This module is tested on Python 3.8.17


```bash
python main.py SOURCE_VIDEO_PATH TARGET_VIDEO_PATH --select_road_lines_by_gui --update_lines_in_config --select_ref_points_by_gui --update_ref_points_in_config --resize_video_by_half
```
### CLI Arguments:
- select_road_lines_by_gui : Give this parameter if you use a new video or you haven't created a config for video which includes lane coordinates.
- update_lines_in_config : Save the selected lane coordinates for the video in a config.json named as {source_video_name}_config.json
- select_ref_points_by_gui: Select 4 ROI for automatically selecting reference points. Make sure you selected a a static region where moving objects can't appear in, such as cars or tree leaves. cv2.goodFeaturesToTrack method is used to select a good reference points on these 4 ROIs. Then optical flow of these 4 reference points helps us to create a spatial transformation matrix to stabilize the lane borders.
- update_ref_points_in_config: Save the selected reference points in config.
- resize_video_by_half: Set this if you want to perform inference on a video that exceeds your screen bounds.


### 📈  Inference Speed Comparison of Pytorch vs ONNX Model
- **YOLOv5s6** model was able to perform inference at **4-5 FPS only**, when using CPU.
- Same model converted to **ONNX** performed inference at **20 FPS**, which means approximately **4x** faster inference.

### 📈  Result Visualizations
- For the visual analysis of car direction states over time for given videos, see the [notebook](./visualize_results.ipynb)

### 🛠️  TODO
* [] A more comprehensive object tracking Framework [Norfair](https://github.com/tryolabs/norfair/tree/master) might be leveraged for obtaining more accurate and less switching car tracks. Norfair also uses Optical Flow info for better association of detection bounding boxes between consequtive frames.
* [] A detection postprocess can be used for improving car detection and tracking in distance, such as [sahi](https://github.com/obss/sahi). But this approach might reduce the inference speed dramatically.
* [] Instead of using car bbox bottom center coordinates, we can use a better reference (e.g. car license plate if we have a model capable of detecting them). Because when the car is too near, that passing from bottom of the bridge where camera is mounted, its bounding box's center no more a good reference to estimate the car's current lane state. The projection of the detection bounding box's center onto the road falls onto a different lane, especially for long vehicles.