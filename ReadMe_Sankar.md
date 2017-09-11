## Vehicle Detection Project
---

The goals of this project are as follows:

* Train a classifier to predict cars on road
* Extract and evaluate classifier performance on features like Histogram of Gradients (HOG), color transformation and spatial binning
* Create a pipeline to detect cars on test images
* Test the pipeline on a video with boxes indicating location of real cars

[//]: # (Image References)
[image1]: ./examples/HOGdemo.PNG
[image2]: ./examples/FalsePositives.PNG
[image3]: ./examples/heat.PNG
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

Now that the classifier has been trained to a good accuracy, the next step is to test it on sample images. A sliding window the size of the training image (64x64) is scanned across the test image with some degree of overlap. The classifier predicts if the window contains a car or not. A simple check that can be done is to limit the window search only across a region of the image where cars can realistically exist. 

For example, the test image is of size 1240x720. The window search was restricted by imposing a Y-limit of 400 to 656. The output of the classifier applied on the images are shown below. 

![alt text][image2]

As seen in the above image, the classifier does a nice job on predicting cars but there are also some false positives detected. In order to tackle these false positives, a couple of different strategies are used. The first and most important strategy is the concept of "adding heat" to a box.

**Heat - rejecting false positives**

As seen in the image above, windows are drawn around region where the classifier predicts a car to be present. These are the "hot windows" within the image. The "heat" logic iterates through the different hot windows and adds a weighting factor to pixels that are located within the window. As seen above, there are multiple hot windows overlapping around the location of the car. The pixels that intersect between these hot windows automatically have their heat value increased compared to other locations in the image. It is unlikely that false positives have multiple windows overlapping. By setting a threshold value for the heat in a given window, false positives are greatly reduced. 

```sh
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap
```
Image below demonstrates the effectiveness of the technique. As seen, all the false positives in the test images were eliminated. 

![alt text][image3]




















