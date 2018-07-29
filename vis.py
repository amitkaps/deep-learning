import numpy as np
import pandas as pd

import altair as alt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.preprocessing.image import ImageDataGenerator


# Helper to get the labels for each class of Fashion Mnist
def fashion_mnist_label(): 
    labels = {
        0: "T-shirt/top", 1:"Trouser", 2:"Pullover", 3:"Dress",  4:"Coat", 
        5:"Sandal",  6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot"}
    return labels


# Show an image from an nparray
def imshow(X, label="", colormap="greys"):
    """
    Shows an image output from a numpy input
    
    X : array_like, shape (n, m)
    label: A string value for the label of the image (default is blank)
    colormap: A color map scheme to apply to the image (default is "grey")
    
    """
    img = pd.DataFrame(X).reset_index().melt("index")
    img.columns = ["h" , "w", "value"]
    image = alt.Chart(img).mark_rect().encode(
        alt.X('w:N', axis=None), 
        alt.Y('h:N', axis=None), 
        alt.Color("value", legend=None, sort="descending", scale=alt.Scale(scheme=colormap)),
        tooltip = ["value"]
    ).properties(
        width = 350,
        height = 350,
        title = label
    )
    return image

# Show first unique value from numpy dataset 
def imshow_unique(X, y, labels):
    """
    Shows an image output from a numpy input
    
    X : array_like, shape (count x width x height)
    y : array_like, shape (count)
    labels: A dictionary of labels for y
    
    """
    u, indices = np.unique(y, return_index=True)
    plt.figure(figsize = (16,7))
    for i in u:
        plt.subplot(2,5,i+1)
        plt.imshow(X[indices[i]], cmap="gray")
        plt.title(labels[y[indices[i]]])
        plt.axis('off')
        
# Create a sprite from the numpy dataset
def imshow_sprite(X):
    """
    Returns a sprite image consisting of images passed as argument. 
    Images should be count x width x height
    """
    if isinstance(X, list):
        X = np.array(X)
    img_h = X.shape[1]
    img_w = X.shape[2]
    n_plots = int(np.ceil(np.sqrt(X.shape[0])))
     
    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))
    
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < X.shape[0]:
                this_img = X[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img
    
    plt.figure(figsize = (10,10))
    plt.imshow(spriteimage,cmap='gray')
    plt.axis('off')        
        
        

# Plot a 3d 
def plot3d(X,Y,Z):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, color='y')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
        
# Visualise the metrics from the model
def metrics(history):
    df = pd.DataFrame(history)
    df.reset_index()
    df["batch"] = df.index + 1
    df = df.melt("batch", var_name="name")
    df["val"] = df.name.str.startswith("val")
    df["type"] = df["val"]
    df["metrics"] = df["val"]
    df.loc[df.val == False, "type"] = "training"
    df.loc[df.val == True, "type"] = "validation"
    df.loc[df.val == False, "metrics"] = df.name
    df.loc[df.val == True, "metrics"] = df.name.str.split("val_", expand=True)[1]
    df = df.drop(["name", "val"], axis=1)
    
    base = alt.Chart().encode(
        x = "batch:Q",
        y = "value:Q",
        color = "type"
    ).properties(width = 300, height = 300)

    layers = base.mark_circle(size = 50).encode(tooltip = ["batch", "value"]) + base.mark_line()
    chart = layers.facet(column='metrics:N', data=df).resolve_scale(y='independent')    
    
    return chart


def predict(proba, actual, labels):
    """
    Shows a probability output from an probability run
    
    proba : array of probability for each class
    actual: an int for the actual class
    labels: a dictionary of labels for each class
    
    """
    df = pd.DataFrame({"proba": proba})
    df['labels'] = df.index
    df['labels'] = df['labels'].map(labels)
    df["actual"] = df.index
    df.loc[df.index == actual, "actual"] = True
    df.loc[df.index != actual, "actual"] = False
    predicted_class = df.proba.idxmax()
    
    chart = alt.Chart(df).mark_bar().encode(
        alt.X('proba:Q', scale=alt.Scale(domain=[0,1])), 
        alt.Y('labels:N'), 
        alt.Color("actual"),
        tooltip = ["proba"]
    ).properties(
        width = 350,
        height = 350,
        title = "Prediction: " + labels[predicted_class]
    )
    return chart


def show_images(images, labels):
    """
    Shows the set of batch image output from a numpy input
    
    images : A set of images with count * width * height * channel
    index: An index for the label for the categorical images
    
    """
    num = len(images)
    columns = 5
    rows = num//5
    i = 0
    plt.figure(figsize = (16,7))
    for img in images:
        plt.subplot(rows,columns,i+1)
        plt.imshow(img)
        label = "label=" + str(labels[i])
        plt.title(label)
        plt.axis('off')
        i = i + 1
        
def show_single_image_gen(gen, image, num):
    """
    Shows the set of image augmented images for a single image
    
    gen: generator object for image augemtation
    image: image to be augmented
    num: number of augmented images
    
    """
    image_array = np.expand_dims(image, axis=0)
    gen.fit(image_array)
    samples = gen.flow(image_array)
    
    images = samples.next()
    for i in range(num-1):
        img = samples.next()
        images = np.r_[images, img]
    
    columns = 5
    rows = num//5
    i = 0
    plt.figure(figsize = (16,7))
    for img in images:
        plt.subplot(rows,columns,i+1)
        plt.imshow(img)
        plt.axis('off')
        i = i + 1 