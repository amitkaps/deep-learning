import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.examples.tutorials.mnist import input_data
import os

# Helper to get the labels for each class of Fashion Mnist
def fashion_mnist_label(): 
    labels = {
        0: "T-shirt/top", 1:"Trouser", 2:"Pullover", 3:"Dress",  4:"Coat", 
        5:"Sandal",  6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot"}
    return labels


# Helpers to create a sprite image in the Embedding Projector
def create_sprite(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    
    
    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))
    
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img
    
    return spriteimage


# Helpers to get a sample of images in the Embedding Projector
def create_embedding(data, name, sample):
    """
    To get a sample of image tensors in to tensorboard embedding projector
    
    data: the dataset to create the projection
    name: the name of the image embedding
    sample_count: How many samples to take from the entire dataset
    """
    
    # Create path and file names
    data_path = "data/" + data + "/"
    log_path = "logs/" + name + "/"
    sprite_file = name + ".png"
    path_for_sprites =  os.path.join(log_path, sprite_file)
    path_for_metadata =  os.path.join(log_path,'metadata.tsv')
    
    # Read the data
    inputs = input_data.read_data_sets(data_path, one_hot=False)
    batch_xs, batch_ys = inputs.train.next_batch(sample)
    #batch_xs, batch_ys = x_train[:sample], y_train[:sample] 
    
    # Create the embedding variable and summary writer
    embedding_var = tf.Variable(batch_xs, name=name)
    summary_writer = tf.summary.FileWriter(log_path)
    
    # Configure the embedding projector
    projector = tf.contrib.tensorboard.plugins.projector
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    
    # Specify where you find the metadata
    embedding.metadata_path = 'metadata.tsv'
    
    # Specify where you find the sprite (we will create this later)
    embedding.sprite.image_path = sprite_file
    embedding.sprite.single_image_dim.extend([28,28])
    
    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, config)
    
    # Create sprite
    to_visualise = 1 - np.reshape(batch_xs,(-1,28,28))
    sprite_image = create_sprite(to_visualise)
    plt.imsave(path_for_sprites,sprite_image,cmap='gray') 
    
    # Run TensorFlow to create the variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(log_path, "model.ckpt"), 1)  
    

    
    # Create metadata
    #open(path_for_metadata, 'a').close()
    with open(path_for_metadata,'w') as f:
        f.write("Index\tLabel\n")
        for index,label in enumerate(batch_ys):
            f.write("%d\t%d\n" % (index,label))
    
    # Print the run instructions
    print("""
    Created embedding in the directory -> %s 
    Run the following command from the terminal
    
    tensorboard --logdir=%s""" % (log_path, log_path))
    
    
# Helper for saving batch-wise 
class MetricHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))

        
        
# Helper to plot prediction
def plot_prediction(index, x_test, y_test, input_data, model):
    label_array = ["T-shirt/top", "Trouser", "Pullover", "Dress",  "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    proba = model.predict_proba(input_data)
    fig = plt.figure(figsize=(4, 6)) 
    plt.subplot(211)
    plt.imshow(x_test[index], cmap="gray")
    plt.title(label_array[y_test[index]])
    
    plt.subplot(212)
    plt.barh(y=range(len(proba[index])), width=proba[index], tick_label=label_array)
    plt.xlim(0,1)
    
    plt.tight_layout()
    
    
# Helper to plot 2d models
def plot_2d_model(model, x, y):

    # Calculate the Classification Boundaries
    x1_min, x1_max = x[:,0].min(), x[:,0].max()
    x2_min, x2_max = x[:,1].min(), x[:,1].max()
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, (x1_max - x1_min)/100), 
        np.arange(x2_min, x2_max, (x2_max - x2_min)/100))
    Z = model.predict_classes(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    
    # Set the 2d points
    plt.figure(figsize=(16,6))
    cmap = plt.cm.get_cmap('plasma', 10)
    
    plt.subplot(121)
    scatter = plt.scatter(x = x[:,0], y = x[:,1], c = y, s = 0.5, cmap=cmap, alpha = 0.3)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    
    plt.subplot(122)
    cs = plt.contourf(xx1, xx2, Z, cmap=cmap, alpha = 0.6)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.clim(0, 9)
    
    # Format the colorbar
    label_array = ["T-shirt/top", "Trouser", "Pullover", "Dress",  "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    ticks = range(len(label_array))
    formatter = plt.FuncFormatter(lambda val, loc: label_array[val])
    plt.clim(0, 9)
    plt.colorbar(scatter, ticks=[0,1,2,3,4,5,6,7,8,9], format=formatter)