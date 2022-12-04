# TikTokProject

## Background
This project was born out of frustration with TikTok's search engine. The TikTok search engine searches for videos based on hashtags and ranks them based 
views and likes. Thus, if a certain search phrase describes the video accurately but does not match the hashtags that the video is tagged with, the video
would be impossible to find.

## Solution
My solution was to develop a machine learning algorithm that automatically tags videos with hashtags. I chose to execute this using a video classification 
algorithm developed by Facebook AI Research, as outlined in [Multiscale Vision Transformers](https://arxiv.org/abs/2104.11227) and [MViTv2: Improved Multiscale Vision Transformers for Classification and Detection](https://arxiv.org/abs/2112.01526)

## Model Details

As the papers highlight, this is a transformer based model that scales videos down, randomly samples clips from them, and feeds them into the model. The 
model itself is a layer of convolution, followed by a layer of pooling, followed by a Multi Head Pooling Attention (a modification to Multi Head Attention 
outlined in the legendary paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) ), which feeds into the fully connected output layer. 

## Dataset

I obtained the data I used to train my model with using the [TikTokAPI by David Teather](https://github.com/davidteather/TikTok-Api). The only data I obtained from this model was publicly available videos and the hashtags they were tagged with. Currently, I only have 31 different hashtags in my model, with an average of 100 videos per hashtag. This means that given a TikTok video, my model will label the video with one of these 31 labels. In the future, I plan on dramatically increasing the number and variety of labels, which will lead to my model being able to label videos with more accurate hashtags.

## Impelementation

I used an Amazon S3 bucket to store all of my data in and connected it to an instance of Amazon Sagemaker, where I trained my model and can perform predictions on videos using the model.

## Codebase

Most of this code is from the [SlowFast](https://github.com/facebookresearch/SlowFast) set of models developed by Facebook Research. I wrote a script 
to split my data, create a PyTorch Dataset class so that I can feed my data into the model.

## Reflections

The most important thing I learned was the inner workings of CNNs, RNNs and Transformers. I started this project by researching Convolutional and Recurrent Neural Networks, as I thought that they would be the best fit for my project. Through this, I gained a better understanding of both architectures, as well as their problems (especially the Recurrent Neural Network's vanishing/exploding gradient problem). This gave me an appreciation as to why Transformers are so widely used and why they are so revolutionary. This also helped me understand how Transformers work.

The model is written in Python using the PyTorch package, and writing code and tracing the [SlowFast](https://github.com/facebookresearch/SlowFast) code has given me a deep understanding of PyTorch. My experience using a the [TikTokAPI by David Teather](https://github.com/davidteather/TikTok-Api) gave me a better understanding of data scraping and best practices to use for scraping, as well as the ethics of scraping. The [Multiscale Vision Transformers](https://arxiv.org/abs/2104.11227) model came with many configurations with the number of parameters ranging from 23 million to 667 million. Thus, I learned to make a decision about the tradeoff between accuracy and computation time, given the resources I have available to me. Since this was the first time I was working with Amazon Web Services, I also learned how to set up and connect Amazon Sagemaker and Amazon S3, and how to use the services effectively to train my model.

## Next Steps

I want to focus on deploying the model. First, I want to automate the task of obtaining new data and training the model on this new data periodically. I will then work on creating a website interface where users can upload .mp4 videos and get a hashtag prediction from my model. 
