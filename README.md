# Automated Annotations Project Overview

Below is a rough outline of the development process for this project.

0. Problem Statement: A YouTube channel was looking to automate a portion of their workflow. They create reviews/summaries of comics. The visual elements of their videos are images of cropped panels from the page of the comic they're summarizing. The objective here was to take in a sample page and crop it in consonance with their style. They provided the training data, which consisted of annotations they created in CVAT and the images. Here's a basic example randomly selected from the validation set:

<table>
    <tr>
        <th>Ground Truth</th>
        <th>Prediction</th>
    </tr>
    <tr>
    <th> <img src="example/example_truth.png" width="286" height="3475"/>  </th>
    <th> <img src="example/example_recall.png" width="286" height="3475"/> </th>
    </tr>
</table>


1. Proof of concept and preparation: After visualizing a few of the annotations at random, I started with the preparation. The data needed a fair amount of cleaning: the annotations were exported in a JSON that contained bad path names and the image subdirectories weren't structured correctly. The JSON format is also incomaptible with YOLO, so it was reformatted. After fixing this, I  trained a minimal model and showed it to the channel owners. 

2. Training: After getting feedback from the channel owners, I trained the entire model. The model is YOLO via Ultralytics. It runs object detection inference on a given image and generates bounding boxes for objects it detects. Currently, there are only two classes (panel and background) although this is likely going to change to cover certain edge cases.
	
3. Containerization: I implemented the model as a web-app with Flask. It takes in an uploaded file and prompts a download for a .zip file containing the full annotated image and crops. I containerized with Docker. In the Dockerfile, the web-app is configured to Gunicorn. 

4. Deployment: I pushed the Docker image to a public repository on AWS. I created a task to run an instance of the image on an ECS cluster with a Fargate capacity provider. 