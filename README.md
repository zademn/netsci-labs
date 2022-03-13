# Network science 2021

**Note**
- [nbviewer](https://nbviewer.jupyter.org/github/zademn/netsci-labs/tree/master/) might have prettier math and image rendering than github
- Photos are taken from the resources (most of them from Stanford's cs224w and Graph representation learning – William Hamilton 2020) or the extra resources linked in the notebooks.

## Prerequisites 

1. Programming: OOP, being able to read and change functions to do something else.
2. Some math: Probabilities, statistics and linear algebra
3. Graph theory – data structures

## Split  
Part 1. Labs 1-5 -- Traditional generative methods, community detection.  
Part 2 -- ML on graphs -- Embeddings, Classification, GNNs

## Environment
Either make an anaconda env or a venv and install the requirements (Please do this before the lab)

Some of the used libraries: `networkx`, `karateclub`, `pytorch`, `pytorch_geometric`

## Colab links

| Lab  	| Link 	|
|:---:	|:---:	|
| 1 Intro |Linalg recap [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/netsci-labs/blob/master/Lab1-Intro/Linalg_recap.ipynb) <br>networkx tutorial short [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/netsci-labs/blob/master/Lab1-Intro/networkx_tutorial_short.ipynb) <br>probs recap [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/netsci-labs/blob/master/Lab1-Intro/probs_recap.ipynb) <br>python tutorial [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/netsci-labs/blob/master/Lab1-Intro/python_tutorial.ipynb) <br>tutorial [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/netsci-labs/blob/master/Lab1-Intro/tutorial.ipynb) <br>| 
| 2 Graph measurements |Graph measurements [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/netsci-labs/blob/master/Lab2-Graph-measurements/Graph_measurements.ipynb) <br>| 
| 3 Random_network_models |Random graphs [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/netsci-labs/blob/master/Lab3-Random_network_models/Random_graphs.ipynb) <br>| 
| 4 Small worlds |Small worlds [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/netsci-labs/blob/master/Lab4-Small-worlds/Small_worlds.ipynb) <br>| 
| 5 Communities |Communities [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/netsci-labs/blob/master/Lab5-Communities/Communities.ipynb) <br>| 
| 6 DL intro |PyG intro [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/netsci-labs/blob/master/Lab6-DL-intro/PyG_intro.ipynb) <br>Torch intro [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/netsci-labs/blob/master/Lab6-DL-intro/Torch_intro.ipynb) <br>| 
| 7 Embeddings |Node embeddings [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/netsci-labs/blob/master/Lab7-Embeddings/Node_embeddings.ipynb) <br>| 
| 8 Node classification |Node classification [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/netsci-labs/blob/master/Lab8-Node-classification/Node_classification.ipynb) <br>| 
| 9 Graph neural networks |GNN intro [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/netsci-labs/blob/master/Lab9-Graph-neural-networks/GNN_intro.ipynb) <br>| 
| 10 Graph neural networks |GNN2 Training [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/netsci-labs/blob/master/Lab10-Graph-neural-networks/GNN2_Training.ipynb) <br>GNN2 Training deepsnap [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/netsci-labs/blob/master/Lab10-Graph-neural-networks/GNN2_Training_deepsnap.ipynb) <br>Graph prediction [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/netsci-labs/blob/master/Lab10-Graph-neural-networks/Graph_prediction.ipynb) <br>Link prediction [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/netsci-labs/blob/master/Lab10-Graph-neural-networks/Link_prediction.ipynb) <br>| 
| 11 Knowledge graphs |Knowledge graphs [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/netsci-labs/blob/master/Lab11-Knowledge-graphs/Knowledge_graphs.ipynb) <br>TransE [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zademn/netsci-labs/blob/master/Lab11-Knowledge-graphs/TransE.ipynb) <br>| 

## Datasets

[Stanford's Snap](http://snap.stanford.edu/data/index.html)

[Network Repository](http://networkrepository.com/)

[Open Graph Benchmark](https://ogb.stanford.edu/), [ogb paper](https://arxiv.org/pdf/2005.00687.pdf)

[TUDataset](https://chrsmrrs.github.io/datasets/)

[Relational dataset repository](https://relational.fit.cvut.cz/)

[moleculenet](https://moleculenet.org/datasets-1)

## Books, courses and more resources

[Network science – Albert-Laszlo Barabasi](http://networksciencebook.com/)

Stanford course cs224w -- Big recommendation
- [yt playlist - Jure Leskovec](https://www.youtube.com/watch?v=JAB_plj2rbA&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn)
- [stanford cs224w – Jure Leskovec](https://web.stanford.edu/class/cs224w/)

[A course in network science](http://www.leonidzhukov.net/hse/2021/networks/)

[Graph representation learning – William Hamilton 2020](https://www.cs.mcgill.ca/~wlh/grl_book/)

[A new kind of science – Stephen Wolfram](https://www.wolframscience.com/nks/)

[Networks, Crowds, and Markets: Reasoning About a Highly Connected World -- By David Easley and Jon Kleinberg)](http://www.cs.cornell.edu/home/kleinber/networks-book/)

[GNNPapers](https://github.com/thunlp/GNNPapers#survey-papers)

[Pytorch Geometric Tutorial](https://github.com/AntonioLonga/PytorchGeometricTutorial)

### Articles
- https://distill.pub/2021/gnn-intro/
- https://distill.pub/2021/understanding-gnns/

## TODO 
Stuff that I need to finish (not exhaustive)
- [ ] install links for jupyter colab

