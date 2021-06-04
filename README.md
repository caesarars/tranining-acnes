# Image Classification on Acnes problem using VGG16
dataset acquaired from https://www.kaggle.com/data/58249
## Data Preprocessing
Our objective is to classify acnes, but the problems is they are too small in the images as you can see inside the red box below:

[![acne-pustular-15-boxed.jpg](https://i.postimg.cc/SxVfdTKm/acne-pustular-15-boxed.jpg)](https://postimg.cc/gxXhkKR7)

We believe it's gonna be real problem, and make the model so poor. so we decide to crop the images so it only show the acnes

[![acnes-pustular-cropping.jpg](https://i.postimg.cc/Jz1zmK5D/acnes-pustular-cropping.jpg)](https://postimg.cc/QHy3JQPh)

we've collected the data and **got about 160 images each**. I think it's not enough for the training so I do **Data Augmentation** by rotating all the images by 10 Degree anti-clockwise up to 90 Degree

[![aug-acnes-pustular-cropping.jpg](https://i.postimg.cc/tCBvQ25T/aug-acnes-pustular-cropping.jpg)](https://postimg.cc/PP8Q1b0s)
