#**Traffic Sign Recognition** 



[//]: # (Image References)

[image1]: ./writeup/dataSplit.jpg "Visualization"
[image2]: ./writeup/rgb.jpg        "RGB Image"
[image3]: ./writeup/architecture.jpg        "RGB Image"
[image4]: ./writeup/convL1.jpg        "Layer 1 Visualization"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image8]: ./examples/placeholder.png "Layer 1 Visualization"

[sign01]: ./webImages/01.jpg           "Traffic Sign - Stop"
[sign02]: ./webImages/02.jpg           "Traffic Sign - Stop"
[sign03]: ./webImages/03.jpg           "Traffic Sign - Speed Limit 30"
[sign04]: ./webImages/04.jpg           "Traffic Sign - Speed Limit 50"
[sign05]: ./webImages/05.jpg           "Traffic Sign - Bumpy Road"

[sign01]: ./webImages/large/01.jpg           "Traffic Sign - Caution"
[sign02]: ./webImages/large/02.jpg           "Traffic Sign - Stop"
[sign03]: ./webImages/large/03.jpg           "Traffic Sign - Speed Limit 30"
[sign04]: ./webImages/large/04.jpg           "Traffic Sign - Speed Limit 50"
[sign05]: ./webImages/large/05.jpg           "Traffic Sign - Bumpy Road"

 

---
### Writeup 


This  Writeup covers the rubric points and describes how each point is addressed.

The project files are located to my GitHub profile [project code](https://github.com/hakimka/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Summary of the data set

To obtain the basic numbers about the data, I used using python and numpy.

The following parameters were calculated:

* The size of training set = 27839
* The size of test set = 12630
* The size of validation set = 6960
* The shape of a traffic sign image = (32, 32, 3)
* The number of unique classes/labels in the data set = 43


####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is split between training, test, and validation sets.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to keep the images in RGB schema.

Here is an example of a traffic sign image.

![alt text][image2]

The amount of data present in the data sets was sufficient So, I decided to use the split of the data as mentioned above for training the Neural Net:



####2. Model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.



![alt text][image3]
My final model consisted of the following layers:

| Layer         		|     Description	        					| 

#### Input 
*    32x32x3 RGB image
 ----
  
####Layer 1

*     Convolution 3 x 3 x 10 , stride 1    produces 32x32x10 tensor
*     RELU                                 produces 32x32x10 tensor  
*     Pool Max    kernel 2 x 2, stride 1   produces 32x32x10 tensor

####Layer 2

*     Convolution 3 x 3 x 6 , stride 1    produces  32x32x6 tensor
*     RELU                                 produces 32x32x6 tensor  
*     Pool Max    kernel 2 x 2, stride 1   produces 32x32x6 tensor

####Layer 3

*     Convolution 5 x 5 x  6, stride 1     produces 28x28x6 tensor
*     RELU                                 produces 28x28x6 tensor 
*     Pool Max   kernel 2 x 2, stride 2    produces 14x14x6 tensor

####Layer 4

*     Convolution 3 x 3 x  16, stride 1    produces 12x12x16 tensor
*     RELU                                 produces 12x12x16 tensor 
*     Pool Max   kernel 2 x 2, stride 2    produces 6x6x16tensor

####Layer  - 
*     Flattened     	                   produces 576x1 tensor

####Layer  - Fully Connected 1
*     Matrix Multiply + Bias              produces 576x120 tensor
*     Relu                                

####Layer  - Fully Connected 2
*     Matrix Multiply + Bias              produces 120x120 tensor
*     Relu                                

####Layer  - Fully Connected 3
*     Matrix Multiply + Bias              produces 120x100 tensor
*     Relu                                 

####Layer  - Logit
*     Matrix Multiply + Bias              produces 43x1 tensor


####3. Model Training. 
To train the model, I used an Adam optimizer for cross entropy between the answer set and the produced results. The answer set was presented as one_hot vector. The learning rate is 0.001. There were 50 epochs. During each epoch a batch size was set to 1024. 

####4. Approach taken for finding a solution
 and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of 85%
* validation set accuracy of 96% 
* test set accuracy of 86%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Testing the Model on New Images

####1. German traffic signs found on the web.
 
Here are four German traffic signs that I found on the web:

![alt text][sign01]![alt text][sign02] ![alt text][sign03] 
![alt text][sign04] ![alt text][sign05]

For each image, I had to bring the size of it to 32x32 in RGB format. 



####2. Discuss the model's predictions 

All signs got identified properly.

Here are the results of the prediction:

 * General Caution Sign - Stop Sign 
 * Stop Sign            - Stop Sign
 * 30 km/h              - 30 km/h
 * 50 km/h              - 50 km/h
 * Bumpy Road           - 30 km/h

The model was able to correctly guess 5 of the 5 known traffic signs, which gives an accuracy of 100%. 

####3. Softmax probabilities for each prediction

If we consider how certain the model is on its prediction we see the following:


For the "General Caution"   the probabilities as follows:
		
	      Probability    Sign Index
		  1.00000000e+00    18   
		  8.38737054e-17    26     ,   
		  1.87590772e-31    27          
		  0.00000000e+00     0   
		  0.00000000e+00     1


For the "Stop Sign"   the probabilities as follows:

For the "General Caution"   the probabilities as follows:
		
	      Probability    Sign Index
		  1.00000000e+00    14   
		  3.08485442e-12    17     ,   
		  1.36946381e-14    0          
		  3.62502286e-23    32   
		  2.99219738e-23     1


For the "30 km/h"   the probabilities as follows:
          
          Probability    Sign Index
		  8.48612785e-01    1   
     	  1.51387155e-01	2	  
		  9.02051840e-17    4   
		  5.46193562e-19    7
		  5.39442646e-21    5
The 30 km/h sign has the probability of 84%. The next to it stand the sign of 50 km/h speed limit with 14%.

For the "50 km/h"   the probabilities as follows:

* 100%   50 km/h
* 
          Probability    Sign Index
		  1.00000000e+00    2   
     	  1.36936148e-11	1	  
		  1.56706516e-13    4   
		  1.64790800e-25    3
		  1.64360680e-26    8

In case of 50 km/h sign, it came with 100% certainty.

	
For the "Bumpy Road"   the probabilities as follows:


          Probability    Sign Index
		  9.99885082e-01    22   
     	  1.14950693e-04	29	  
		  9.01706722e-15    30  
		  9.01706722e-15    28
		  9.01706722e-15    25


.  

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

When I provided image of "50 km/h" sign to the trained net and visualized the convoluted/relu'd/maxpooled layers. I saw this image:

![alt text][image4]

It appears that the filter Feature Map 2 picked up the numbers inside the sign. Where as the Feature Map 4 picked up the round red border from the sign. 