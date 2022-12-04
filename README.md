# TikTokProject

## Background
This project was born out of frustration with TikTok's search engine. The TikTok search engine searches for videos based on hashtags and ranks them based 
views and likes. Thus, if a certain search phrase describes the video accurately but does not match the hashtags that the video is tagged with, the video
would be impossible to find.

## Solution
My solution was to develop a machine learning algorithm that automatically tags videos with hashtags. I chose to execute this using a video classification 
algorithm developed by Facebook AI Research, as outlined in [MViT](https://arxiv.org/abs/2104.11227) and [MViTv2](https://arxiv.org/abs/2112.01526)

## Implementation

As the papers highlight, this is a transformer based model that scales videos down, randomly samples clips from them, and feeds them into the model. The 
model itself is a layer of convolution, followed by a layer of pooling, followed by a Multi Head Pooling Attention (similar to Multi Head Attention 
outlined in the legendary paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) ). 
