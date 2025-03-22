# Image Machine Learning (IML) Project

The Image Machine Learning (IML) Project is a suite of Python tools designed to streamline various aspects of image processing and analysis using machine learning techniques. Each component addresses a specific task—ranging from image cropping and extraction to ranking and aggregation—ensuring a comprehensive and flexible workflow for visual data pipelines.

## Components

1. **[IML Cropper](https://github.com/royokello/iml-cropper)**  
   A tool to train and apply an image cropping model. It predicts crop boxes and performs batch cropping on images within a directory.

2. **[IML Extractor](https://github.com/royokello/iml-extractor)**  
   A utility for extracting a specified number of frames from all videos in a directory. Supports both sequential and random selection for flexible video frame analysis.

3. **[IML Ranker](https://github.com/royokello/iml-ranker)**  
   A system that ranks images using pairwise comparison and Elo rating. It collects user preferences, extracts features, and trains models to predict ranking behavior.

4. **[IML Cull](https://github.com/royokello/iml-cull)**  
   A filter for removing low-quality or unwanted images from datasets. Works well as a post-processing step after ranking or cropping.

5. **[IML Aggregator](https://github.com/royokello/iml-aggregator)**  
   A script that recursively collects images and their captions from nested directories, converts images to PNG, and saves them in a unified output directory.
