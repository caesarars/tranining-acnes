# Image Classification on Acnes problem using VGG16
Dataset acquaired from https://www.kaggle.com/data/58249 
Dataset that we use https://drive.google.com/drive/folders/1jhIrDNXshobSyKxjZqmuamnbz1F7mDZI?usp=sharing

## Data Preprocessing
Our objective is to classify acnes, but the problems is they are too small in the images as you can see inside the red box below:

[![acne-pustular-15-boxed.jpg](https://i.postimg.cc/SxVfdTKm/acne-pustular-15-boxed.jpg)](https://postimg.cc/gxXhkKR7)

We believe it's gonna be real problem, and make the model so poor. so we decide to crop the images so it only show the acnes

[![acnes-pustular-cropping.jpg](https://i.postimg.cc/Jz1zmK5D/acnes-pustular-cropping.jpg)](https://postimg.cc/QHy3JQPh)

We choose only two kind of acnes that are pustular and excoriated. Why? because those acnes are the most clearer object that computer would identify after do convolution on them.
1. Cystic 
    - [![cystic-acne-305.jpg](https://i.postimg.cc/mDh7sXPT/cystic-acne-305.jpg)](https://postimg.cc/DJkSQcQN)
2. Pustular  
    - [![pustular-7.jpg](https://i.postimg.cc/02SdS90F/pustular-7.jpg)](https://postimg.cc/ykVZ445y)
3. Excoriated  
    - [![acne-excoriated-25.jpg](https://i.postimg.cc/C5qDb4kk/acne-excoriated-25.jpg)](https://postimg.cc/303dTmLx)

we've collected the data and **got about 160 images each**. 
I think it's not enough for the training so I do **Data Augmentation** by rotating all the images by 10 Degree anti-clockwise up to 90 Degree

[![aug-acnes-pustular-cropping.jpg](https://i.postimg.cc/tCBvQ25T/aug-acnes-pustular-cropping.jpg)](https://postimg.cc/PP8Q1b0s)
