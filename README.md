# Herarchical-LSTM
Implementation of the paper: Rare Entity Prediction with Hierarchical LSTMs Using External Descriptions

The paper link: [http://aclweb.org/anthology/D17-1086]

The dataset can be obtained from the following link: [http://dataset.cs.mcgill.ca/downloads/rare_entity_dataset.html]

The dataset consists of two files
* corpus.txt: contains paragraphs containing blanks. Each blank stores the id of the corresponding correct entity.
* entities.txt: contains the entity id followed by the entity name and description of the entity.

**read_data.py**

This file parses the the two files and organises the data into a single file to be fed into the model. It also creates a dictionary for the words seen in the data.


**network.py**

This file consists of the model. It contains the code to read from the filed created by read_data.py and run the model for the given number of epochs.

I have used pre-trained glove embeddings for training and also kept them trainable during the actual training of the model.

The glove embeddings can be obtained as follows:

```
wget http://nlp.stanford.edu/data/glove.6B.zip 
unzip glove.6B.zip -d content
```

The code still has to be organised and commented to make it more clear.


