# Image Machine Learning (IML) Project

The Image Machine Learning (IML) Project comprises a suite of Python tools tailored for comprehensive image processing and analysis. Each tool addresses a specific aspect of the image preparation pipeline, from data aggregation to quality assessment, cropping, and ranking.

## Components

1. **[IML Aggregator](https://github.com/royokello/iml-aggregator)**  
   A Python script that retrieves all image files from a specified directory and its subdirectories, converts them to PNG format, and saves them to a designated output folder. Additionally, the script collects any caption text files that share the same filename as the images.

2. **[IML Extractor](https://github.com/royokello/iml-extractor)**  
   A Python library designed to efficiently extract a specified number of frames from all videos within a given directory. It supports both sequential and random frame selection, making it ideal for tasks such as video analysis, machine learning, and data preprocessing.

3. **[IML Cull](https://github.com/royokello/iml-cull)**  
   A simple image culling toolkit with manual labeling built on Flask, AI-assisted culling recommendations based on Google ViT, and automated image management.

4. **[IML Cropper](https://github.com/royokello/iml-cropper)**  
   A Python app for training an image cropping model, predicting crop boxes, and batch cropping images in a directory.

5. **[IML Ranker](https://github.com/royokello/iml-ranker)**  
   A Python app for ranking images using machine learning and the Elo rating system. It collects user preferences via pairwise comparisons, extracts image features, and trains models to predict preferences, enabling efficient identification of top-rated images.

## Suggested Workflow for Preparing Images for Training

To effectively prepare images for training purposes, consider the following sequence:

1. **Data Collection**:  
   Use [IML Aggregator](https://github.com/royokello/iml-aggregator) to gather and standardize images from various directories, or [IML Extractor](https://github.com/royokello/iml-extractor) to capture frames from video files.

2. **Quality Assessment**:  
   Apply [IML Cull](https://github.com/royokello/iml-cull) to review and filter the collected images, removing those that are unsuitable or of low quality.

3. **Image Cropping**:  
   Use [IML Cropper](https://github.com/royokello/iml-cropper) to generate cropped versions of the images with your desired aspect ratio.

4. **Image Ranking**:  
   Use [IML Ranker](https://github.com/royokello/iml-ranker) to evaluate and rank the processed images based on learned user preferences.
