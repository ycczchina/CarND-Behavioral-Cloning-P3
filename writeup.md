# **Behavioral Cloning** 

### ***Zheng Chen***

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png "Model Visualization"
[image2]: ./examples/center_2018_10_03_00_02_17_230.jpg "center lane driving example"
[image3]: ./examples/center_2018_10_03_00_03_45_497.jpg "Recovery Image"
[image4]: ./examples/center_2018_10_03_00_03_46_327.jpg "Recovery Image"
[image5]: ./examples/center_2018_10_03_00_03_31_000.jpg "Recovery Image"
[image6]: ./examples/center_2018_10_03_00_03_46_327.jpg "Normal Image"
[image7]: ./examples/center_2018_10_03_00_03_46_327_flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 24 and 64 (model.py lines 107-111) 

The model includes RELU layers to introduce nonlinearity (code line 107-111), and the data is normalized in the model using a Keras lambda layer (code line 105). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 114, 116). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 121). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 120).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving. I used pictures of center, left and right cameras and corresponding angles. I also flipped the images and angles since I only ran the track counter-clockwise.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to search a powerful neural network and modify it for the project.

My first step was to use a convolution neural network model similar to the NVIDIA CNN architecture I thought this model might be appropriate because the goals of both projects are similar.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model with two dropout layers so that the performance will be improved.

The final step was to run the simulator to see how well the car was driving around track one. There were 2 spots where the vehicle fell off the track. To improve the driving behavior in these cases, I added more side back to center data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 104-118) consisted of a convolution neural network with the following layers:

1. Keras lambda layer to normalize the data
2. Cropping2D layer to select image section
3. 3 convolutional layers with 5x5 kernel and 2x2 stride. RELU is also applied
4. 2 convolutional layers with 3x3 kernel. RELU is also applied
5. Flatten layer
6. 4 fully-connected layers and first two have a dropout of 0.5

Here is a visualization of the architecture (from NVIDIA CNN)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to keep away from both sides. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would help the training since I only ran it counter-clockwise. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 10170 data points. I then preprocessed this data by normalization.I finally put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.

And following is the generated video:

```
<a href="http://www.youtube.com/watch?feature=player_embedded&v=TNRbon5s5NI
" target="_blank"><img src="http://img.youtube.com/vi/TNRbon5s5NI/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>
```

```
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/TNRbon5s5NI/0.jpg)](http://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID_HERE)
```