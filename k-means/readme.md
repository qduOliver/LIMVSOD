In k-means, run .py sequentially in numerical order.

Through the OB section, we get all the object boxes。
first,1proportion.py changing the value of the motion value according to the significance proportion of the optical flow frame occupied by each object boxes.
Second, 2k_means.py cluster all the object boxes into 8 classes. 
Third,3julei_MV ,evaluate a motion value value for each class. 
Then,4, 5 and 6 are to divide the 8 clusters into positive and negative samples according to the motionvalue value.
Then,7motionValue_choose.py and 8second_julei.py filter positive and negative samples by motion value and distance to the center of the class, respectively. 
Finally 9 got the data used to train the classifier for the first time.
