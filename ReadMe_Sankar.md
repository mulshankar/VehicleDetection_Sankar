##Vehicle Detection Project

---

The goals of this project are as follows:

* Train a classifier to predict cars on road
* Extract and evaluate classifier performance on features like Histogram of Gradients (HOG), color transformation and spatial binning
* Create a pipeline to detect cars on test images
* Test the pipeline on a video with boxes indicating location of real cars

[//]: # (Image References)
[image1]: ./examples/HOGdemo.PNG
[image2]: ./examples/FalsePositives.PNG
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

**Description of files**

* VehicleDetectionSankar.ipynb is the jupyter notebook that contains the overall code
* FinalProjectVideoOutput.mp4 is the video output demonstrating the above pipeline

**Algorithm**
---

The first step was to gather data for training a classifier that can detect cars. The GTI and KITTI vehicle and non-vehicle databases were used for training purposes. The total number of car images were 8792 and non-car images were 8968. It is important that the ratio is close to 1 so that there is no bias in the classifer. 

The next step was to select features for the classification problem. The 3 key features that were used for the classification were:

- Spatial domain
- Color transformation and
- Histogram of Gradients (HOG)

HOG features offer a clear distinction between cars and non-car images. HOG feature use edge detection and computes a histogram of gradients within a user-defined block. Image below shows a comparison of HOG feature on a car image vs non-car image. 

![alt text][image1]

**Building the classifier**

A simple linear Support Vector Machine (SVM) classifier was chosen for the detection problem. Various combinations of color, spatial and HOG features were iterated upon to quantify the classification accuracy. 

- In terms of spatial feature, a 32x32 image was sufficient to capture the key features of a car
- Various color spaces were explored that include HSV, HLS, LUV, YUV and YCrCb. In terms of classifier accuracy, HLS and YCrCb performed the best. It was interesting to note that LUV and YUV features caused some NaN values while extracting HOG features on some images. The final color space chosen for this project was YCrCb
- HOG features were extracted from all 3 channels. It made the feature vector slightly longer but helped a lot in terms of classification accuracy. 

The final parameter set chosen for the linear SVM was:

	```sh
		color_space='YCrCb' # HSV, HLS, LUV, YUV, YCrCb
		spatial_size=(32,32)
		hist_bins=32 
		orient=9 
		pix_per_cell=8 
		cell_per_block=2 
		hog_channel='ALL' # 1,2,'ALL'                   
		spatial_feat=True 
		hist_feat=True 
		hog_feat=True
	```
The total time taken to extract features from all the cars and non-cars images was 184.4 seconds. The length of feature vector was 8460. The features were dumped into a pickle file to facilitate SVM tuning later without having to extract features repeatedly. The feature vector was scaled using StandardScaler function available via sklearn. 

	```sh
		# Use a linear SVC 
		svc = LinearSVC()
		# Check the training time for the SVC
		t=time.time()
		svc.fit(X_train, y_train)
		t2 = time.time()
		print(round(t2-t, 2), 'Seconds to train SVC...')
		# Check the score of the SVC
		print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

		# save classifer to disk
		with open("svc_pickle.p", "wb" ) as file:
			pickle.dump(svc, file)
	```
	
The linear SVM took 20.11 seconds to train. The overall accuracy of the SVM was 98.85% which was deemed sufficient. Similar to feature extraction, the classifier parameters were dumped into a pickle file to be used later. 

**Window Search**

Now that the classifier has been trained to a good accuracy, the next step is to test it on sample images. A sliding window the size of the training image (64x64) is scanned across the test image. The classifier predicts if the window contains a car or not. A simple check that can be done is to limit the window search only across a region of the image where cars can realistically exist. 

For example, the test image is of size 1240x720. The window search was restricted by imposing a Y-limit of 400 to 656. The output of the classifier applied on the images are shown below. 

![alt text][image2]

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

