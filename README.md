# GCN-SeA
Repository for the TACL 2019 paper: "Graph Convolutional Network with Sequential Attention For Goal-Oriented Dialogue Systems"

## Abstract
Domain specific goal-oriented dialogue systems typically require modeling three types of inputs, *viz*., (i) the knowledge-base associated with the domain, (ii) the history of the conversation, which is a sequence of utterances and (iii) the current utterance for which the response needs to be generated. While modeling these inputs, current state-of-the-art models such as Mem2Seq typically ignore the rich structure inherent in the knowledge graph and the sentences in the conversation context. Inspired by the recent success of structure-aware Graph Convolutional Networks (GCNs) for various NLP tasks such as machine translation, semantic role labeling and document dating, we propose a memory augmented GCN for goal-oriented dialogues. Our model exploits (i) the entity relation graph in a knowledge-base  and (ii) the dependency graph associated with an utterance to compute richer representations for words and entities. Further, we take cognizance of the fact that in certain situations, such as, when the conversation is in a code-mixed language, dependency parsers may not be available. We show that in such situations we could use the global word co-occurrence graph and use it to enrich the representations of utterances. We experiment with 4 datasets, *viz.*, (i) the modified DSTC2 dataset (ii) recently released code-mixed versions of DSTC2 dataset in four languages (iii) Wizard-of-Oz style CAM676 dataset and (iv) Wizard-of-Oz style MultiWOZ dataset. On all the 4 datasets our method outperforms existing methods, on a wide range of evaluation metrics.

## Requirements
spacy 2.0 <br />
tqdm <br />
nltk 3.4 <br />
tensorflow 1.8 <br />
logging <br />
scipy 1.1 <br />

## Data Pre-processing 
We need to pre-process the raw dialogue text files into dictionaries containing the dependency/contextual graphs for dialogue history, query and create the Knowledge graph from the KB triples. All these are done by ```preprocess_dstc2.py``` for the English and code-mixed versions of DSTC2 dataset. <br />
* ### Pre-process En-DSTC2:
   ```python preprocess_dstc2.py --source_dir ../data/en-dstc2 --target_dir ../data/en-dstc2 --rand_edges False --use_associations False ``` <br />
* ### Pre-process Code-Mixed versions with ppmi used for contextual graphs:
   ```python preprocess_dstc2.py --source_dir ../data/hi-dstc2 --target_dir ../data/hi-dstc2 --rand_edges False --use_associations True --dict_name edge-dict-ppmi-dstc2```

## Training and Inference
To train the model on En-DSTC2 run:
   ```python main.py -train=True -config_id=1 -data_dir=../data/en-dstc2 -data_name=/preprocessed-dialog-dstc2-final_data.json -rnn_unit=gru -edge_gate=True -use_rnn=True -learning_rate=0.0008 -l2=0.001 -batch_size=32 -epochs=30 -num_hops=1 -dropout=0.9 -word_emb_dim=300 -rnn_hidden_units=300 -gcn_hidden_units=300 -seed=1234 -init=trunc```

* To train on code-mixed versions of DSTC2 change ```-data_dir``` and ```-data_name``` appropriately.
* To run in Inference mode change ```-train=False```
* To run Inference on provided trained models ensure that ```-config_id``` matches with the checkpoint directory number. For example, use ```-config_id=35``` for running Inference on provided model for En-DSTC2.

