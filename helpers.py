import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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