## Vehicle Detection Project
---

The goals of this project are as follows:

* Train a classifier to predict cars on road
* Extract and evaluate classifier performance on features like Histogram of Gradients (HOG), color transformation and spatial binning
* Create a pipeline to detect cars on test images
* Test the pipeline on a video with boxes indicating location of real cars

[//]: # (Image References)
[image2]: ./examples/OrientationDOE.png
[image3]: ./examples/PixPerCellDOE.png
[image4]: ./examples/CellPerBlockDOE.png
[image1]: ./examples/HOGdemo.PNG
[image5]: ./examples/FalsePositives.PNG
[image6]: ./examples/HeatSample.PNG
[image7]: ./examples/heat.PNG
[image8]: ./examples/HeatMapThreshold.PNG
[video9]: ./FinalProjectVideoOutput.mp4

**Description of files**

* VehicleDetectionSankar.ipynb is the jupyter notebook that contains the overall code
* FinalProjectVideoOutput.mp4 is the video output demonstrating the above pipeline

**Algorithm**
---

The first step was to gather data for training a classifier that can detect cars. The GTI and KITTI vehicle and non-vehicle databases were used for training purposes. The total number of car images were 8792 and non-car images were 8968. It is important that the ratio is close to 1 so that there is no bias in the classifer. 

The next step was to select features for the classification problem. The 3 key features that were used for the classification were:

- Spatial
- Color transformation and
- Histogram of Gradients (HOG)

HOG features are a real powerful tool for the classification problem and offered a clear distinction between cars and non-car images. HOG features use edge detection and computes a histogram of gradients within a user-defined block. Image below shows a comparison of HOG feature on a car image vs non-car image. 

![alt text][image1]

**Building the classifier**

A simple linear Support Vector Machine (SVM) classifier was chosen for the detection problem. Various combinations of color, spatial and HOG features were iterated upon to quantify the classification accuracy. 

- In terms of spatial feature, a 32x32 image was sufficient to capture the key features of a car
- Various color spaces were explored that include HSV, HLS, LUV, YUV and YCrCb. In terms of classifier accuracy, HLS and YCrCb performed the best. It was interesting to note that LUV and YUV features caused some NaN values while extracting HOG features on some images. The final color space chosen for this project was YCrCb
- HOG features were extracted from all 3 channels. It made the feature vector slightly longer but helped a lot in terms of classification accuracy. A Design of Experiments (DOE) was performed on the HOG parameters. The first parameter that was studied was the HOG orientations. Image below shows the effect of number of orientations for a car image. As evident from the image below, increasing orientations beyond 9 did not add any value. When the number of orientations was set at 9, the a clear pattern was noticed for car image.  

![alt text][image2]

The next HOG parameter that was studied was the "pixel_per_cell". Image below shows the effect of this parameter on HOG features for car vs non-car images. At a value around 8, the hog image seemed to have a clear pattern surrounding the car. 

![alt text][image3]

The "cell_per_block" parameter did not seem to have a major effect on the final hog image. 

![alt text][image4]

Eventually all these combinations were evaluated by calculating the accuracy of the SVM classifier using these features. Accuracy had to be >98% to control false positives on the image. 

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

Now that the classifier has been trained to a good accuracy, the next step is to test it on sample images. A sliding window the size of the training image (64x64) is scanned across the test image with some degree of overlap. The classifier predicts if the window contains a car or not. A simple optimization was done to limit the window search across a region of the image where cars can realistically exist. 

For example, the test image is of size 1240x720. The window search was restricted by imposing a Y-limit of 400 to 656. The output of the classifier applied on the images are shown below. 

![alt text][image5]

As seen in the above image, the classifier does a nice job on predicting cars but there are also some false positives detected. In order to tackle these false positives, a couple of different strategies were used. The first and most important strategy was the concept of "adding heat" to a box.

**Heat - rejecting false positives**

As seen in the image above, windows are drawn around region where the classifier predicts a car to be present. These are the "hot windows" within the image. The "heat" logic iterates through the different hot windows and adds a weighting factor to pixels that are located within the window. As seen above, there are multiple hot windows overlapping around the location of the car in an image. The pixels that intersect between these hot windows automatically have their heat value increased compared to other locations in the image. It is unlikely that false positives have multiple windows overlapping. By setting a threshold value for the heat in a given window, false positives are greatly reduced. 

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
Image below shows the heat map that was generated. As seen, the ones that are brighter tend to be around the location of the car 

![alt text][image6]

Below is the test image with false positives eliminated via the heat map strategy. 

![alt text][image7]

While the sliding window search is effective, it is computationally very expensive. Another scheme was written to extract hog features only once and then sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor where a scale of 1 would result in a window that's 8 x 8 cells. 

**Final Pipeline**

While the heat method worked great on test images, false positives were still detected on the project video. Also, the box was very jittery around the car. Few additional strategies were used to improve noise rejection and stabilize the bounding box around the car.

- An aspect ratio check was added to hot windows. If the detected window in an image is too small to realistically be a car, those windows were automatically eliminated. 
- A running buffer of heat map was maintained for the last 10 frames. The sum (or average) of this heatmap buffer was used for thresholding. This greatly helped to stabilize the bounding box around the car. 
- Multiple scales were used to parse the image for cars. For example, smaller scales were used for detection of cars that were farther in the image and larger scales for cars that are closer. 
- Rather than using SVM classifier to output a 0 or 1 for car/non-car, the SVM decision function was used. The decision function produced the distance of the current "X - test feature" from the classifier boundary. In other words, it provided a confidence value of how sure the classifier is about the windows containing a car or not. A threshold was set for this boundary and proved extremely useful in reducing the false positives.

The final pipeline for the image processing is shown below:

```sh
def final_pipeline(image_jpg):
    
    global heatmaps
    global heatmapsumlist

    hotwindows=find_cars(image_jpg)
    
    heat = np.zeros_like(image_jpg[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,hotwindows) 
    #print(np.max(heat))
    heatmaps.append(heat)
    #print(heatmaps)
    heatmap_sum = sum(heatmaps)
    heatmapsumlist.append(np.max(heatmap_sum))
    #print(np.max(heatmap_sum))
    #heatmaps_avg=get_avg_heatmaps(heatmaps,10)
    #print(np.max(heatmaps_avg))

    # Apply threshold to help remove false positives
    heat = apply_threshold(heatmap_sum,30)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image_jpg), labels)
    
    return draw_img
```

In order to get a reasonable estimate of the threshold, the heatmap sum was plotted for the region of the video that was challenging with shadows and detected more false positives. A sample plot of the heat sum is shown below. As seen, a threshold of 30 worked well across the whole video.  

![alt text][image8]


The pipeline was tested on the project video. A link to the video is below (youtube)

[![Final Video](https://img.youtube.com/vi/2g5UC7db_Eg/0.jpg)](https://youtu.be/2g5UC7db_Eg)


**Scope for Improvement**

* Combination of color spaces might help in better noise rejection
* A deep learning based approach might be interesting to benchmark in comparison




















