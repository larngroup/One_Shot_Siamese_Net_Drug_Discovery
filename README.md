# One-shot Siamese Neural Network Architecture for Drug Discovery
The application of deep neural networks is an important asset to significantly increase the predictive power when inferring the properties and activities of small-molecules and those of their pharmacological analogues. However, in the traditional drug discovery process where data is very scarce, the lead-optimization step is, inherently, a low-data problem, which makes it difficult to find analogous molecules with the desired therapeutic activity, making it impossible to obtain accurate predictions for candidate drug analogues.
\
\
One major requirement to validate the obtained neural network models and one of its biggest limitations is the need for a large number of training examples per class which, in many applications, might not be feasible. This invalidates the use of instances whose classes were not considered in the training phase or in data where the number of classes is high and oscillates dynamically.
\
\
The main objective of this paper is to optimize the discovery of novel compounds, with increased therapeutic activity for the same pharmacological target, based on a reduced set of candidate drugs. We propose the use of a Siamese neural network architecture for one-shot classification, based on Convolutional Neural Networks (CNNs), that learns from a similarity score between two input molecules according to a given similarity function.
\
\
Using a one-shot learning strategy, we only need a few instances per class for the network's training and a small amount of data and computational resources to build an accurate model.
\
\
The results of this study show that a one-shot-based classification using a Siamese neural network allowed us to outperform the graphical convolution and the state-of-the-art models in the accurate and reliable prediction of novel compounds given the lack of biological data available for candidate molecules in drug discovery tasks.

# Our Approach
In this study, starting from a reduced set of candidate molecules, we adapt the convolutional neural network (CNN) to predict novel compounds according to the structural dissimilarities observed between molecules. The proposed model accepts different pairs of molecules and learns a similarity function which returns a similarity score between two input molecules. Thus, according to the learned similarity rule the network predicts the similarity score in one shot.
\
\
We introduce an approach compatible with the set of pairs of compounds provided, a Siamese neural network built upon two parallel and identical convolutional neural networks. The model learns a similarity function and returns a distance metric applied to the output feature vectors from both Siamese twins given a pair of input molecules. 

# Methodology
## Model Overview

We introduce a model that accepts data organized in pairs, a Siamese neural network - two parallel and identical convolutional neural networks (CNNs). Both Siamese twins are indistinguishable, being two copies of the same network, sharing the same set of parameters. 
\
These parallel networks reduce their respective inputs to increasingly smaller tensors as we progress to high-level layers. The difference between the output feature vectors is used as an input to a linear classifier, a similarity function. 

## Siamese Neural Network

The main model consists of a Siamese Neural Network based on convolutional neural networks (CNNs), which uses a single sequence of layers with filters of varying size and number. The network's input consists of two twin input vectors with a corresponding hidden vector in each of the convolutional layers. The model architecture that maximizes performance is the one whose number of convolutional layers is 3, whose number of filters in each layer is a multiple of 16 and in which the corresponding output features maps are applied to a ReLu activation function and to a maxpooling layer. The application of a dropout layer at the end inactivates a portion of the neurons preventing overfitting.
\
\
The output feature map of the last convolutional layer is flattened into a single vector that serves as an input to a fully connected layer with 1024 units.
\
\
This layer learns a similarity function between two feature vectors by applying a distance metric to the learned feature map. 
Then, it is followed by a dense layer that computes the absolute difference between the two output feature maps. 
This value serves as input to a sigmoid function in the last layer.

## Model Architecture



## Pairwise Training

Since the siamese net accepts pairs of molecules, the dataset size increases, as the number of possible combinations for the pairs of molecules for training increases by a quadratic factor. 
\
However, since we have to consider half of pairs of the same class and half of different classes in training, the total number of possible pairs is limited by the number possible pairs of the same class.
\
The number of training instances increases in of a square factor, preventing overfitting.

## One-Shot Learning Approach

The reduced amount of biological data for training led us to adopt a new strategy to predict novel compounds using the proposed model. We consider a one-shot classification strategy to demonstrate the discriminative power of the learned features.
\
Note that for every pair of input twins, our model generates a similarity score between 0 and 1 in one shot. Therefore, to evaluate whether the model is really able to recognize similar molecules and distinguish dissimilar ones, we use an N-way one shot learning strategy. The same molecule is compared to N different ones and only one of those matches the original input. Thus, we get N different similarity scores {p_1, ..., p_N} denoting the similarity between the test molecules and those on the support set. We expect that the pair of compounds with the same class gets the maximum score and we treat this as a correct prediction. If we repeat this process across multiple trials, we are able to calculate the network accuracy as the percentage of correct predictions.
\
\
In practice, for model validation we select a test instance that is compared to each one of the molecules in the support set. The support set consists of set of molecules representing each class selected at random whenever a one-shot learning task is performed.
\
Subsequently, we pair the test instance with each of the compounds in the support set and check which one gets the highest similarity score according to the learned similarity rule. We conclude that the prediction is correct if the maximum score corresponds to the pair of molecules of the same class, that is, if the test instance and the instance from the support set are from the same class. Thus, in each trial, we organize the pairs for validation so that the first pair is a pair of instances of the same class, with the remaining pairs formed by compounds of different classes.


# Model Comparison and Results

The comparison of a given complex model with a set of simpler base models is a common strategy when assessing performance. In this case, compounds are represented by matrices so it was necessary to reduce their dimension by converting them to pairs of flattened vectors. This representation led to a consistent model evaluation and to a meaningful performance comparison between different models.
\
\
In the implementation of those models, we provide a  set  of  drug  pairs and we merge and represent them as concatenated flattened vectors. We consider a training set in which  half  are  pairs  of  the  same  class  and  another  half  of different classes. To ensure consistency in the performance evaluation,  the  accuracy  of  the  model  was  determined using  the  same  strategy mentioned previously.  We  define  a  set  of  N-way  one-shot  tasks  that  allows  the comparison of a set of  N concatenated pairs across 500 trials. The  prediction p is  correct  if  the  concatenated  pair  of  the same class obtains the highest value.

## Accuracy Results


| Model | #2-way  | #3-way  | #4-way | #5-way  | #7-way  | #10-way |
| ------- | --- | --- | --- | --- | --- | --- |
| Siamese val | 95% | 90% | 86% | 78% | 70% | 60% |
| Siamese train | 94% | 92% | 84% | 86% | 72% | 62% |
| KNN | 69% | 55% | 48% |  42% | 35% | 35% |
| Random | 61% | 43% | 34% | 31% | 22% | 10% |
| SVM | 56% | 42% | 24% | 30% | 16% | 12% |
| Random Forest | 71% | 58% | 60% | 44% | 34% | 20% |
| CNN |  81% | 70% | 58% | 46% | 41% | 39% |
| MLP | 76% | 60% | 36% | 34% | 22% | 13% |

# Future Work

With this work, we conclude that it is possible to carry out a more rigorous study by increasing the N value in the N-way testing phase, increasing the number of trials and adjusting the hyperparameters, allowing us to achieve more accurate predictions and to prevent overfitting.
\
On the other hand, we aim to validate how one-shot classification using Siamese Neural Networks enable us to achieve strong results given the lack of biological data available for candidate compounds in drug discovery tasks.
