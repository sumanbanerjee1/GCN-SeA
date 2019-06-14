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
