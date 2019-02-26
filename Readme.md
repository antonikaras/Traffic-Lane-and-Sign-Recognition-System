# Traffic Lane and Sign Recognition System

## Description

	This app can be used to detect Lanes and Signs from a video or 	a camera. It's written in python and 
	uses OpenCV library.

## Context

	* signs: Sign database used by SURF and ORB detector
	
	* svm: Trained SVM data
	
	* ORBdetector.py: App that compares an image with the sign database and returns what sign it is, or 
			  -1 if it's not a sign using ORB detector.
	
	* SignRecognition.py: App that detects potential signs in an image/video and recognises which sign it is.

	* SURFdetectpr.py: App that compares an image with the sign database and returns what sign it is, or 
			   -1 if it's not a sign using SURF detector.

	* SVMdetector.py: App that compares an image with the sign database and returns what sign it is, or 
			  -1 if it's not a sign using a SVM classifier.

	* TrafficLaneRecognition.py: App that detects the lanes in the road.

	* TrafficMarkingsRecognition.py: App that uses TrafficLaneRecognition.py and SignRecognition to detect 
					 the lanes and signs in a Video.

## Dependencies
	
	* Python, Tested on Python 3
	* OpenCV3, Tested on opencv 3.4.2.16
	* numpy
	* scipy
	* scikit-learn

## Use 
	
	You can use the Lane and Sign detector seperately running the .py app or the combined system running 
	TrafficMarkings app. For each app you use, you need you have to provide a video or a camera with 
	resolution 1280*720.
