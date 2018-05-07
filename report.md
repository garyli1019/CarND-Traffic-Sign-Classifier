# **Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./internet_images/1.jpg
[image2]: ./internet_images/2.jpg
[image3]: ./internet_images/3.jpg
[image4]: ./internet_images/4.jpg
[image5]: ./internet_images/5.png
[image8]: ./internet_images/result.png
[image9]: ./internet_images/histogram.png

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/lyj19940105/Udacity-CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.
![alt text][image9]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Step 1: Convert the image into grayscale by rgb2gray function. Because we are using convolution net work and only care about the feature in the image, so the color images and grayscale images would perform almost the same. Grayscale images were smaller size and less number of layer, which will significantly reduce memeory requirement size and improve training speed.
Step 2: Normalize the image by (pixel - 128)/128. By normalizing all the pixels, the training images will be more relate to each other and the entire dataset will be consistent.
Step 3: In order to make the mean close to 0, I added an extra function that pixel = (pixel - mean(pixels)). In this case, the mean is 0.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    |  1x1 stride, valid padding, outputs 10x10x16     									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten	      	| input: 5x5x16 		output: 400		|
| Fully connected		| input: 400   output: 120        									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| input: 120   output: 84        									|
| RELU					|												|
| Dropout					|												|
| Fully connected		| input: 84   output: 43        									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

batch size=128
learning rate = 0.001
epochs = 100
I used this parameters because the Lenet lecture provided those. I have tested couple different parameters, like 50 epochs and 0.0005 learning rate, but the accuracy rate was lower than this. I think the 0.0005 was a little bit too low when epochs number was a small number. When we want to train our model with large epochs number, we need to consider reduce learning rate overtime to avoid overfiiting problem.
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.6
* validation set accuracy of 94.7
* test set accuracy of 92.4

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The architecture I used was Lenet with RGB images. The accuracy rate was not high enough and I thought the speed could be improved by changing the images to grayscale.
The second attemp was Lenet with gracscale images. Tried couple different parameters but the model wasn't great enough to achieve 93% accuracy rate and have some overfitting issue.

* What were some problems with the initial architecture?
Accuracy rate not high enough, training speed was slow, overfitting issue

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I inserted two dropout layers after the activation layers in the fully connected layers session. Dropout layer eliminated the overfitting issue and improved the accuracy rate.

 

### Test a Model on New Images

#### 1. Choose seven German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]

image 5 is difficult to classify, because when resize the image into 32x32, it was hard to distinguish even with human eye. It looks like "!" sign.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

![alt text][image8]

The images from internet are real life images, so we need to do more preprocessing before the classfication. Even the accuracy rate was high on the test set, it doesn't mean that the algorithm will work well in the real life.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

See above image


