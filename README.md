# Deep Learning Bootcamp

The objective for the Deep Learning bootcamp is to ensure that the participants have enough theory and practical concepts of building a deep learning solution in the space of computer vision and natural language processing. Post the bootcamp, all the participants would be familiar with the following key concepts and would be able to apply them to a problem.

**Key Deep Learning Concept**
- **Theory**: DL Motivation, Back-propagation, Activation
- **Paradigms**: Supervised, Unsupervised
- **Models**: Architecture, Pre-trained Models (Transfer Learning)
- **Methods**: Perceptron, Convolution, Pooling, Dropouts, Recurrent, LSTM
- **Process**: Setup, Encoding, Training, Serving
- **Tools**: python-data-stack, keras, tensorflow

## Notebooks

- **001**: [Theory - Deep Learning, Universal Approximation, MLP for tabular data](/001-Theory-DL.ipynb)
- **002**: [Multi-layer Perceptron - Fashion MNIST](002-MLP-Fashion.ipynb)
- **003**: [Theory - Convolution Neural Network](/003-Theory-CNN.ipynb)
- **004**: [Convolution Neural Network - Fashion MNIST](/004-CNN-Fashion.ipynb)
- **005**: [Transfer Learning - Fashion MNIST](/005-Transfer-Learning-Fashion.ipynb)
- **006**: [Data Augmentation - Fashion MNIST](/006-Data-Augmentation-Fashion.ipynb)
- **007**: [MLP & CNN - Dosa/No Dosa](/007-MLP-CNN-DosaNoDosa.ipynb)
- **008**: [Data Augmentation - Dosa/No Dosa](/008-Data-Aug-DosaNoDosa.ipynb)
- **009**: [Transfer Learning - Dosa/No Dosa](/009-Transfer-Learning-DosaNoDosa.ipynb)
- **010**: [Theory & Concept - Natural Language Processing](010-NLP-Basics.ipynb)
- **011**: [Recurrent Neural Network - Toxic Classification](/011-RNN-LSTM-Toxic.ipynb)
- **012**: [Convolution - 1D - Toxic Classification](/012-CNN-1D-Toxic.ipynb)
- **013**: [Pre-Trained Embedding - Words - Toxic Classification](/013-PreTrained-Words-Toxic.ipynb)
- **014**: [Pre-Trained Embedding - Sentences - Toxic Classification](014-PreTrained-Sentence-Toxic.ipynb)

## Learning Resources by Authors

- Presentations
  - [Practical Guidance for Deep Learning](/dl-practical-guidance.pdf)
  - [Deep Learning For Image](https://speakerdeck.com/amitkaps/deep-learning-for-image)
  - [Deep Learning For NLP](https://www.slideshare.net/amitkaps/deep-learning-for-nlp-69972908) 
- Long Form Articles 
  - [Logo Detection](https://www.oreilly.com/ideas/logo-detection-using-apache-mxnet)  by @bargava
  - [Uncovering Hidden Pattern](https://www.oreilly.com/ideas/uncovering-hidden-patterns-through-machine-learning) by @bargava
  - [How to learn Deep Learning in 6 months](https://towardsdatascience.com/how-to-learn-deep-learning-in-6-months-e45e40ef7d48) by @bargava


## External References
These are reference material which have good explanations - visual, interactive, symbolic or code driven in text, video or notebook form - about key concepts in Machine Learning and Deep Learning. We have found them useful in our own learning journey.

- Basics: Python, Numpy and Math
  - Don't know **python**. Start with a crash course from @anadology using [Python Practice Book](https://anandology.com/python-practice-book/), (*Text, Code*)
  - Don't know **numpy**. Start with a good introduction here from @jakevdp at [Section 2. Introduction to Numpy in Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/). (*Notebook, Code*)
  - Want a refresher in **Linear Algebra and Calculus**. Watch these videos by @3blue1brown on [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) and [Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr). (*Video, Visual*)

- Basics of Machine Learning
  - Never done any **Machine Learning**. Start with the first four chapters of [Section 5. Machine Learning in Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/). (*Notebook, Code*)
  - How do you **build, select and validate a Machine Learning Model**. Read these three blogs posts by Sebastian Rashcka on Model evaluation, model selection, and algorithm selection in machine learning: [Part 1 - Basics](https://sebastianraschka.com/blog/2016/model-evaluation-selection-part1.html), [Part 2 - Holdout](https://sebastianraschka.com/blog/2016/model-evaluation-selection-part2.html), [Part 3 - Cross Validation & Hyper Parameter Tuning](https://sebastianraschka.com/blog/2016/model-evaluation-selection-part2.html). (*Text, Math & Visual*)
  - Want to know **the math in ML**? Check out our repo on HackerMath - Learning MAth for ML using code and visuals. [HackerMath](https://github.com/amitkaps/hackermath/). *(Notebook, Visual & Code)*

- Deep Learning Basics
  - Want a **visual understanding of Deep Learning**. Start with these four videos by @3blue1brown on [Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi). *(Video, Visual)*
  - Want to **play & learn how to create a neural network**? Go and play with all the knobs and options to build and train a simple neural network at [Tensorflow Playground](https://playground.tensorflow.org/). *(Website, Interactive)* 
  - How can **neural networks compute any function**? Read this visual proof by Michael Nielson in [Chapter 4 in Neural Network and Deep Learning](http://neuralnetworksanddeeplearning.com/chap4.html). *(Text, Visual)*
  - Why are simple neural networks (like MLP) **hard to train**? Here is a good explanation on the concept of *vanishing* and *exploding* gradients in Deep Learning - [Chapter 5](http://neuralnetworksanddeeplearning.com/chap5.html). *(Text, Visual & Code)*


- Learning & Optimization
  - What is this **Back-Propogation** stuff? Here is easy to understand visual explanation on [The Calculus of Backpropogation](http://colah.github.io/posts/2015-08-Backprop/). *(Text, Visual & Math)*
  - How do **optimizer works**? Start with this fantastic interactive post on Ben Fredrickson on [Numerical Optimization](https://www.benfrederickson.com/numerical-optimization/). *(Text, Visual & Interactive)*
  - What are all these **optimizers**? Read through this exhaustive explanation on SGD and its variants by Sebatian Ruder on [Optimizing Gradient Descent](http://ruder.io/optimizing-gradient-descent/). *(Text, Visual & Math)*
  - Want to learn more about **Stochastic Gradient Descent**? Read through this interactive article on momentum in SGD: [Why Momentum really works](https://distill.pub/2017/momentum/). *(Text, Interative)*
  - Interested in recent improvements in optimisation. Check out more recents article by Sebastian Ruder on [DL Optimisation Trends](http://ruder.io/deep-learning-optimization-2017/) and Fast.ai post on [Adam Weight Decay](http://www.fast.ai/2018/07/02/adam-weight-decay/). *(Text, Math)*
  

- Deep Learning for Images
  - ...
  - ...
  - ...


- Deep Learning for NLP
  - Confused by all these **embedding** stuff? Read this post on [Representation and NLP](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) to understand of why they are so effective in Deep Learning. *(Text, Visual)*
  - Want to understand **word embeddings**? Start with this elegant post on [Word is worth a thousand vectors](https://multithreaded.stitchfix.com/blog/2015/03/11/word-is-worth-a-thousand-vectors/). *(Text, Visual)*
  - How does this **word2vec** stuff relate to **statistical** methods? This article with a click-bait title -  [Stop using word2vec](https://multithreaded.stitchfix.com/blog/2017/10/18/stop-using-word2vec/) will help you put all these methods in a simple framework to understand. *(Text, Visual)*
  - Need to deep dive more in the **math of word embedding**. Start with these four posts by Sebastian Ruder on word embeddings: [Part 1 - Basic](http://ruder.io/word-embeddings-1/), [Part 2 - Softmax](http://ruder.io/word-embeddings-2/),  [Part 3 - Word2Vec](http://ruder.io/secret-word2vec/), [Part 5 - Recent Trends](http://ruder.io/word-embeddings-2017/index.html). *(Text, Math)*
  - What are **RNNs and LSTMs**? Start with this visual unpacking of what is happening within the LSTM node - [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/). **(Text, Visual)**
  - Still confused by all this **DL text approaches**? Here is fantastic article to understand the DL process for NLP as the four steps of **Embed - Encode - Attend - Predict** in this post by Spacy's creator on [Deep Learning Formula for NLP](https://explosion.ai/blog/deep-learning-formula-nlp)
  - Want pratical steps for using Deep Learning for **Text Classification**? Check out how to build a DL model and consolidated best practice advice from [Google's Text Classification Guide](https://developers.google.com/machine-learning/guides/text-classification/step-2-5). *(Text, Visual & Code)*
  - Doing more **exotic NLP** stuff? Then check out this article on current [Best approaches for Deep Learning in NLP] (http://ruder.io/deep-learning-nlp-best-practices/). *(Text, Math)*

- Visualisation
  - Why to we want to **visualise & understand NNs**? This post will give you a core understanding of process of visualising NNs for Human Beings - [Visualising Representation](http://colah.github.io/posts/2015-01-Visualizing-Representations/). *(Text, Visual)*
  - Want to **visualise networks and learning**? Use the [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard) callback to start doing that from your notebooks. *(App, Interactive)*
  - Want to understand why all these **dimensionality reduction** approaches? Start by reading the interactive piece by Christopher Olah on [Visualising MNIST](http://colah.github.io/posts/2014-10-Visualizing-MNIST/) *(Text, Visual & Interactive)*
  - Want to look at **your embeddings in 2D/3D**? Check out the [embedding projector](https://projector.tensorflow.org/) and you can run it on your own data using TensorBoard. *(App, Interactive)*
  - What is the **Neural Network really learning in images**? Check out these articles on [Feature Visualisation](https://distill.pub/2017/feature-visualization/) and [The Building Block of Interpretibility](https://distill.pub/2018/building-blocks/). *(Text & Notebooks, Visual & Interactive)*

- Continue (Your) Learning on (Deep) Learning
  - Want to find more notebooks on Deep Learning?. Explore the collections of interactive ML examples at [Seedbank](https://tools.google.com/seedbank/)
  - More of a book person? My reference for an applied book is this very practical book by François Chollet - [Deep Learning in Python](https://www.manning.com/books/deep-learning-with-python)
