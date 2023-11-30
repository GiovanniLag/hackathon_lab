#  Esperimento Glicerina

#### Abstract

The glycerine viscosity experiment is a rather simple activity which is often part of first-year laboratory courses in a bachelor's degree in physics. The experiment is usually carried out this way: a tall cylinder is filled with the liquid that one would like to study, and some metal spheres are dropped onto the surface of the liquid; the terminal velocity is reached quite quickly and by measuring the elapsed time as the sphere travels a certain pre-imposed distance it is possible to determine the speed value.

By repeating the measurements with balls of different radii it is possible to extract the value of the viscosity after a simple interpolation.

This procedure however often comes with multiple sources of systematic errors which affect the result: the ball may be dropped too close to the cylinder wall, it may be dropped not directly above the surface of the liquid and, more notably, if one uses his/her own eyes to detect if the ball passed at a certain point problems like parallax and slowness of human reflexes may affect the measurement.


During the hackathon we sought to eliminate, or at least reduce, some of these sources of error by designing a new measurement process along with an automatic ball release mechanism with the help of **Arduino, Raspberry Pi and OpenCV**.

In particular, we have implemented a **drop mechanism** that uses an electromagnet to hold the metal ball and release it at the press of a button on the keyboard.
While for the measurements of the falling time we have followed two approaches, first a double camera chronometer and then a single camera **tracking with Kalman filtering** 

## Part 1: release mechanism using an electromagnet

The problems related with the release of the ball have been solved by using an electromagnet which, being mounted on a proper support (Figs below) made it possible to align the magnet with the axis of the cylinder. This way it was possible to be sure that the sphere was always being released at the centre of the liquid surface, so that the additional friction related to the cylinder wall could be avoided.

![](https://cdn.mathpix.com/cropped/2023_11_29_42a5acf52d1c158ad80dg-1.jpg?height=577&width=851&top_left_y=1468&top_left_x=594)

Figure 1: focus on the support and the Arduino processor

![](https://cdn.mathpix.com/cropped/2023_11_29_42a5acf52d1c158ad80dg-2.jpg?height=517&width=654&top_left_y=244&top_left_x=701)

Figure 2: focus on the electromagnet, right above the surface of the liquid

The Arduino processor, to which the electromagnet was attached, was also linked to a separate computer. There a python script, the file ardcontroller.py, was being run, while on the processor the script hackathonlab.ino was loaded so that the release mechanism could be controlled remotely.

## Part 1.1: Problems with the electromagnet configuration

While this system solves the aforementioned biases, there were other challenges which arose. One was that due to the rudimentary support the magnet was prone to overheating. This not only slowed the measurement process, as we had to interrupt ourselves to prevent that, but an increase in temperature may also alter the viscosity of the liquid near the surface, as this parameter is dependent on temperature.
Moreover to make the drop "smooth" we had to raise the level of the glycerine close to the electromagnet, thous making the overheating problem even worse, since we lacked time and resources to create a ventilation / coling system.  
However, these additional effects were not investigated due to the lack of time.

## Part 2.a: Tracking of the falling ball using Kalman Filters
![](https://github.com/GiovanniLag/hackathon_lab/blob/main/readMe_files/result_1.gif)
In the gif above the red circle is the detection while the yellow one is the tracking.

Tracking falling balls requires two steps, detection and tracking, the first refers to the process of identifying the object in the frames and the second involves predicting the ball's position in subsequent frames. 

### Detection
To detect the ball within a frame, our approach involves utilising a pair of frames and calculating the difference between them. Initially, we apply a series of filters to these frames:

```python
# Convert to grayscale and apply Gaussian blur
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (11, 11), 0)
```

Subsequently, we subtract the two consecutive frames. After subtraction, a threshold is applied to the result, followed by a morphological opening operation using OpenCV:

```python
# Compute the frame difference
frame_diff = cv2.absdiff(gray, prev_gray)
_, thresh = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)

# Perform morphological opening
kernel = np.ones((5,5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
```

Finally, we identify contours within the processed frame and focus on the largest one. The centroid of this largest contour is calculated, representing the ball's position:

```python
# Find contours and select the largest one
contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)

# Calculate the centroid of the largest contour
centroid = get_centroid(largest_contour)
```

For more detailed insights, please refer to the `runTracker.py` code.

### Tracking
Kalman Filters are particularly effective for this second step. They provide a means to infer the position of the ball in real time, even in the presence of noise and other uncertainties. The Kalman Filter works by making an initial estimate of the ball's state (including its position and velocity), then updating this estimate as new data (i.e., new frames from the video) becomes available. This process involves two phases: prediction and update. In the prediction phase, the filter uses the current state and a physical model of motion (like Newton's laws of motion) to predict the state at the next time step. In the update phase, the filter incorporates the new observational data to refine its prediction, thereby reducing the uncertainty. By iteratively applying these steps, the Kalman Filter can effectively track the ball as it falls.

In our implementation, the initial position for the Kalman filter is set based on the ball's detection in the first two frames. Subsequent detections are then used to continually adjust and correct the filter's predictions.
(again for further details see `runTracker.py`)

### The distortion problem
Tracking the ball's position accurately presents a significant challenge due to frame distortions. These distortions are caused by two main factors: the high viscosity of glycerin, which alters the path of light through it, and the inherent lens distortion of the Raspberry Pi camera. If not addressed, these distortions could lead to significant inaccuracies in tracking data.

To address this, our script employs a sophisticated approach that mimics an ortho-normal projection to correct for these distortions.

1. **Creating an Ortho-Normal Projection Distortion Profile**:
    
    - The script facilitates the creation of a distortion profile by allowing users to click on pairs of points in a frame. These pairs are points that, in an ideal ortho-normal projection (without distortion), would overlap or align perfectly.
    - When setting `--dist_profile` to "new," the user is prompted to identify these pairs by clicking on the frame: a right click for one point and a left click for its corresponding pair. This process captures the disparity between where a point appears due to distortion and where it should be in an undistorted ortho-normal view.
2. **Interpolation and Adjustment Based on Distortion Profile**:
    
    - The script then uses these pairs of points to develop an interpolation model. This model quantifies the distortion by comparing the actual positions of these points with their expected positions in an undistorted frame.
    - As the script processes new frames, it applies this model to adjust the tracked position of the ball. This adjustment aligns the ball’s apparent position with where it would be in an ortho-normal projection, thus 'correcting' the distortion.

**NOTE:** we are only interested in the vertical distortion since thanks to out dropping mechanism the ball is dropped in the center of the tube; so once we align the camera with the tube center the horizontal distortion is of no concern.
### Implementation
We have implemented all in the script `runTracker.py` which can track the falling ball either from a video or a live camera and it will then save the ball position in a csv file.
#### Usage:
To utilize the script, the following steps and commands should be followed:

1. **Initialization**: The script requires the OpenCV library for video processing and Pandas for data handling. Ensure these libraries are installed in your Python environment.
    
2. **Command Line Arguments**:
    
    - `source`: Specify `0` for using a live camera or `1` for using a video file.
    - `--video_input`: If using a video file, provide the path to the video file.
    - `output`: Specify the path where the output video file will be saved.
    - `--dist_profile`: Optionally, specify the path to a distortion profile file. If not provided, the default profile will be used. Use "new" to create a new profile.
3. **Running the Script**:
    
    - For a live camera feed: `python runTracker.py 0 output_path.csv [--dist_profile dist_profile_path.csv]`
    - For a video file: `python runTracker.py 1 --video_input path_to_video.mp4 output_path.csv [--dist_profile dist_profile_path.csv]`
4. **Distortion Profile**: If "new" is chosen for `--dist_profile`, the script will prompt the user to click on the frame to set up a new distortion profile. This profile is crucial for accurately tracking the ball's position in the distorted medium and also accounts for lens distortion.
    
5. **Output**: The script will save the tracked positions of the ball in a specified CSV file. Additionally, if the distortion profile is set to "new", it will save the new distortion profile in a separate CSV file.
    

#### Notes:

- The script includes real-time plotting capabilities, which are currently commented out due to stability issues.
- For video input, the time calculation might need adjustment to account for the frame rate and processing speed. This can be achieved by using the frame number and video's FPS to calculate the time accurately.

## Part 2.b: detection mechanism with two cameras running in parallel

  
This second approach, though simpler, allows us to completely ignore the distortion problem thanks to the use of two cameras as triggers. Of course cons of these method is that we no longer have all the data points of the ball falling but we just compute the time difference between the ball passing the upper camera and the ball passing the lower camera.
The final configuration, which was reached after two full days of programming, is that reported in Fig. 3-4. A Raspberry Pi is placed on the top of a box and thereto are connected a picamera ("top camera") and a standard webcam ("bottom camera"). This is still a rudimentary configuration, but it served our sake and did not present any new major bias.

![](https://cdn.mathpix.com/cropped/2023_11_29_42a5acf52d1c158ad80dg-2.jpg?height=591&width=926&top_left_y=1760&top_left_x=565)

Figure 3: side view of the cameras configuration

![](https://cdn.mathpix.com/cropped/2023_11_29_42a5acf52d1c158ad80dg-3.jpg?height=643&width=934&top_left_y=244&top_left_x=561)

Figure 4: top view of the cameras configuration

While active the Raspberry Pi was connected to a separate monitor (not shown in the pictures) and it was used to develop the detection mechanism, whose final version is the Python script experiment.py.

An example of the command line used to run the code is below:

```console
python experiment.py 2
```

The only argument is the type of the sphere used, with 1 being the smallest and 4 the greatest. Other numbers are used to indicate test runs rather than actual measure and the data is stored in the file times1.csv. The core of this script is the trigger function, which takes as only parameter a string that indicates which camera to activate (we used "std" for the bottom camera and "" for the top camera).

Once the program runs the function is called twice in parallel, so that both cameras are activated (note lines 208-210). The main idea used to measure the terminal velocity is the following: every frame is converted to a grey scale and the program calculates the difference between two consequent frames; at the price of halving the frame rate this way what does not move appears black, while the objects that move will appear white on the screen.

So as the ball falls through the liquid a white dot is visualized on screen. Once the first camera detects the passage of the sphere at the centre of the screen it saves the current time in the variable time_up (line 190) while when the detection happens at the bottom camera it registers time_down (line 115). The different between these is accounted as the elapsed time and, knowing the distance it is possible to calculate terminal velocity (not included in the script).

In order to be sure that the passage is registered always at the same place the program takes a central stripe of pixels and sums their value; when the ball crosses this section the value of the sum increases and by setting a threshold to this value the time measurement starts/stops when the sphere is right at the centre of the screen (lines 97-105).

### problems with the detection mechanism

This mechanism works quite well with the smaller spheres, while the bigger ones are sometimes too fast to be detected and thus the threshold needs to be increased in that case. Moreover this threshold appeared to be very small in every case, such that even small movements in the surroundings could trigger the cameras (but this can be solved easily by placing the equipment in front of a wall and far from moving objects).

Notwithstanding this the whole procedure has been tested with spheres with radius equal to $1,43 \mathrm{~mm}$ and it appear to work quite well

