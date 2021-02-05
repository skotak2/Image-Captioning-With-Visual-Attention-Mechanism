# Image-Captioning-With-Visual-Attention-Mechanism
Built an encoder - decoder model for captioning an image with visual attention mechanism. Encoding of image is done with CNN and decoding is done with RNN(GRU &amp; LSTM) based networks.

### TABLE OF CONTENTS
* [INTRODUCTION](#introduction)
* [TECHNOLOGIES](#technologies)
* [MODEL-DESIGN](#model-design)
* [ARCHITECTURE](#architecture)
* [CODE](#code)
* [DATA-PRE_PROCESSING](#data-pre_processing)
* [EVALUATION-RESULTS](#evaluation-results)

## INTRODUCTION 
The concept of image captioning is novel problem which deals with cognition of image processing, language modelling and recurrent neural networks. Ability to generate descriptions for image has a myriad number of applications across the industry. The social media platforms have enormous amount of need for these kinds of applications where there is huge surge of images which could help draw insights for business and decision-making purposes. The problem is framed with encoder – decoder flow line with recurrent nueral network layers. Further we also capture the image visual attentions and use it for descriptions. We use pretrained models for implementing CNN as the encoder, and further use RNNs (GRU & LSTM) to translate the objects in image to words to frame a natural sentence. Model takes an image I as input and is trained to maximize the likelihood p(W|I) of producing a target sequence of words W = {W1, W2, W3 . . .Wt} where each word Wt comes from a given vocabulary, while W would also have the start and stop tokens for the sentences. The image vectors are generated using CNN with help of pretrained models. Given the existence of context in the caption a BLEU score metric is used to gauge the performance of model.


## TECHNOLOGIES
Project is created with: 
* Python - **pandas, keras, sklearn, matplotlib,numpy**
* Google - **Colab**


## MODEL-DESIGN
The model has 3 parts for discussion: Encoder, Decoder and Attention. These parts are stitched together for a complete Auto-encoder-decoder model. The encoder model would be a Pre trained CNN nueral network – VGG16, which is build based on ImageNet database. Here we remove the last layer and use the vector representation of the image as an embedding to get an image encoded as a vector. The decoder is built using the RNN network for emitting the required set of words for framing a sentence. Two variants of RNN were involved - LSTM and GRU and different models were built to performing the image captioning task. Attention is generated out of dense nueral network layers to capture the weights of the encoder features and get the focus on that part of the image which needs a caption.

**Encoder:**

The encoder model compresses the image into vector with multiple dimensions. Each element of the vector represents the pixel across different dimension. The VGG16 network represents the image of (224*224*3) into (7*7*512), where the number of pixels is 49. The VGG16 is a novel CNN network with multiple conv layers helping to extract multiple activation maps. The final layer of the VGG16 network is removed and a dense layer with ReLU activation function is used to extract each pixel in 256 dimensions, hence representing the image in (49,256) dimensional vector. This is further used for building the context vector that is used to feed the decoder part of the network.

**Attention Mechanism:**

Attention mechanism is used to compute weights vector that can be used to identify the important hidden states of encoder (pixels of image in our case), and weight those parts of the context more. In our case, we want to focus on those part of the image where we want to focus and describe the
 
gist of the image. The following is the calculation involved in computation, it is as per Bahdanau attention/local attention mechanism

*ht = f(xt)	f = Linear function,	xt = Feature vector obtained from Encoder CNN	(1)*

Next step is using the Decoder LSTM hidden states, we get the internal states corresponding to the hidden states of LSTM.

*hs′=f(ht−1) ht−1=Previous Hidden state obtained from Decoder LSTM	(2)*

Using ht and hs′ we calculate the attention scores i.e. s1, s2, s3, .. sn. The model will learn to relevant encoder states by generating a high score for the states for which attention is to be paid while low score for the states which are to be ignored.

*score = f(relu(ht,hs′)) f = Linear function	(3)*

Upon identifying the scores for images, we use a softmax activation function and calculate the probabilities for these scores i.e. the attention weights e1, e2, e3,..,en.

*e = {e1, e2, e3,..,en} and 0 ≤ e ≤ 1*

*αts=exp(score)Σexp(score) Soft max Activation function is used	(4)*

Using these attention scores, we calculate the context vector which will be used by the decoder in order to predict the next word in the sequence.

*Cv= Σαtsshs′ Cv=Context Vector	(5)*

*St = RNN(St-1, [e(yt-1),Cv]), RNN could be LSTM/GRU where, e(yt-1) is the previous word prediction *

**Decoder:**
A GRU/LSTM which has different time steps, helping to generate sequence of outputs. This sequence helps in building natural sentences, that could be a translated version of the sentence from the other language or description of an image.

## ARCHITECTURE

![GitHub Logo](https://github.com/skotak2/Image-Captioning-With-Visual-Attention-Mechanism/blob/main/Images/Picture1.JPG)


## CODE

We have tried 4 variants in trying different decoder and attention combinations,

1. [Image_Captioning_Decoder_GRU](https://github.com/skotak2/Image-Captioning-With-Visual-Attention-Mechanism/blob/main/Code/Image_Captioning_Decoder_GRU.ipynb)
2. [Image_Captioning_Decoder_GRU_Global_Visual_Attn](https://github.com/skotak2/Image-Captioning-With-Visual-Attention-Mechanism/blob/main/Code/Image_Captioning_Decoder_GRU_Global_Visual_Attn.ipynb) 
3. [Image_Captioning_Decoder_LSTM](https://github.com/skotak2/Image-Captioning-With-Visual-Attention-Mechanism/blob/main/Code/Image_Captioning_Decoder_LSTM.ipynb)
4. [Image_Captioning_Decoder_LSTM_without_visual_attn](https://github.com/skotak2/Image-Captioning-With-Visual-Attention-Mechanism/blob/main/Code/Image_Captioning_Decoder_LSTM_without_visual_attn.ipynb)

## DATA-PRE_PROCESSING
The data for this project is available at [FLICKR8k](https://academictorrents.com/details/9dea07ba660a722ae1008c4c8afdd303b6f6e53b)

We used Flickr_8k dataset, where we process 8000 images with each image having 5 captions. Hence, we train the image 5 times with different captions to make the model sophisticated to predict for the unseen images. Given the image, we get the image to a standard form of (224*224*3), where the three dimensions includer red, green, and blue. These images are further
 
processed through the VGG16 pre trained model to get a encoding of the image in (7*7*512) form, where 512 dimensions are developed using the conv layers. There are further passed through a dense layer to get 256 dimensions encoding.
With captions, we look for the longest caption in the dataset. In our dataset we find 33 worded captions is the longest caption. We have <Start> and <Stop> tags in the captions hence the captions with length less than 33 are padded with zeros. A total of 8329 long vocabulary is built with each word as an index. This is passed through dense layers and the embeddings are retrieved for each word, with dimension of 256.
  
## EVALUATION-RESULTS

**BLEU Score**

The Bilingual Evaluation Understudy is a score for comparing a candidate translation of text to one or more reference translations. A perfect match results in a score of 1.0 whereas a perfect mismatch results in a score of 0. This score is calculated by comparing the n gram of the candidate translation with ngram of the reference translation to count the number of matches.

![GitHub Logo](https://github.com/skotak2/Image-Captioning-With-Visual-Attention-Mechanism/blob/main/Images/Picture2.jpg)

*BLEU scores consists of:*
Brevity penalty (BP): it is to see that high score is assigned to the candidate translation which matches the reference translation in length, word choice and word order.
N: No. of n-grams, we usually use unigram, bigram, 3-gram, 4-gram
wₙ: Weight for each modified precision, by default N is 4, wₙ is 1/4=0.25
Pn: Modified precision score captures two aspects of translation, adequacy and fluency: A translation using the same words as in the references counts for adequacy.
The longer n-gram matches between candidate and reference translation account for fluency

**RESULTS**

Here we see that the attention is very well captured, where the actions, colors and objects are very understood with a good BLEU score of 30+. The LSTM has fairly performed better job in terms of BLEU score, however it was relatively expensive in terms of computation. The attention is well captured here.

LSTM without attention was computational expensive. 40 epochs of training was done against 20 epochs with attention. However, the model without attention underperformed against the one with attention. In our model most of the images were undertrained, only a few images got captioned with one or two words. One of the examples is below. We see that only one word got generated through out without any context involved.

*GRU with local attention*

![GitHub Logo](https://github.com/skotak2/Image-Captioning-With-Visual-Attention-Mechanism/blob/main/Images/Picture3.jpg)


*LSTM with local attention*

![GitHub Logo](https://github.com/skotak2/Image-Captioning-With-Visual-Attention-Mechanism/blob/main/Images/Picture4.jpg)
![GitHub Logo](https://github.com/skotak2/Image-Captioning-With-Visual-Attention-Mechanism/blob/main/Images/Picture5.jpg)

*LSTM without local attention*

![GitHub Logo](https://github.com/skotak2/Image-Captioning-With-Visual-Attention-Mechanism/blob/main/Images/Picture6.jpg)


## REFERENCES
* https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
* https://pytorch.org/tutorials/beginner/saving_loading_models.html

