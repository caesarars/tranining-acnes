# Image Classification on Acnes problem using VGG16
- Dataset acquaired from https://www.kaggle.com/data/58249 
- Dataset that we use https://drive.google.com/drive/folders/1jhIrDNXshobSyKxjZqmuamnbz1F7mDZI?usp=sharing

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

## Training
Load the datasets from Google Drive
```
drive.mount('/content/drive')
 
SOURCE_PATH = "/content/drive/My Drive/ColabNotebooks/acnes_aug/"

train_excoriated_dir = Path(SOURCE_PATH + 'train/excoriated')
train_excoriated = list(train_excoriated_dir.glob(r'**/*.JPG')) + list(train_excoriated_dir.glob(r'**/*.jpg'))

train_pustular_dir = Path(SOURCE_PATH + 'train/pustular')
train_pustular = list(train_pustular_dir.glob(r'**/*.JPG')) + list(train_pustular_dir.glob(r'**/*.jpg'))

train_cystic_dir = Path(SOURCE_PATH + 'train/cystic')
train_cystic = list(train_cystic_dir.glob(r'**/*.JPG')) + list(train_cystic_dir.glob(r'**/*.jpg'))

training_data =  train_excoriated + train_pustular + train_cystic

print("total training data of pustular acne : "+ str(len(train_pustular)))
print("total training data of excoriated acne : "+str(len(train_excoriated)))
print("total training data of cystic acne : "+str(len(train_cystic)))
print("total training data : " + str(len(training_data)))
```

> - total training data of pustular acne : 1600
> - total training data of excoriated acne : 1620
> - total training data of cystic acne : 1600
> - total training data : 4820

Processing image, return the image filepath into DataFrame with the labels
```
def proc_img(filepath):
    """ Create a DataFrame with the filepath and the labels of the pictures
    """

    labels = [str(filepath[i]).split("/")[-2] \
              for i in range(len(filepath))]

    filepath = pd.Series(filepath, name="Filepath").astype(str)
    labels = pd.Series(labels, name="Label")

    # concate the filpaths and labels
    df = pd.concat([filepath,labels], axis=1)

    # shuffle the DataFrame and resetIndex
    df = df.sample(frac=1).reset_index(drop=True)

    return df
```
```
train_df = proc_img(training_data)
print(f'Number of pictures: {train_df.shape[0]}\n')
print(f'Number of different labels: {len(train_df.Label.unique())}\n')
print(f'Labels: {train_df.Label}')
```

### Training 3 Acnes
```
def proc_img(filepath):
    """ Create a DataFrame with the filepath and the labels of the pictures
    """

    labels = [str(filepath[i]).split("/")[-2] \
              for i in range(len(filepath))]

    filepath = pd.Series(filepath, name="Filepath").astype(str)
    labels = pd.Series(labels, name="Label")

    # concate the filpaths and labels
    df = pd.concat([filepath,labels], axis=1)

    # shuffle the DataFrame and resetIndex
    df = df.sample(frac=1).reset_index(drop=True)

    return df
```
