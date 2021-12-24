# LeSICiN
Dataset and codes for the paper "LeSICiN: A Heterogeneous Graph-based Approach for Automatic Legal Statute Identification from Indian Legal Documents", accepted and to be published at AAAI 2022.

# About
The task of **Legal Statute Identification (LSI)** aims to identify the legal statutes that are relevant to a given description of facts or evidence of a legal case.
Existing methods only utilize the textual content of facts and legal articles to guide such a task. However, the citation network among case documents and legal statutes is a rich source of additional information, which is not considered by existing models. 
In this work, we take the first step towards utilising both the text and the legal citation network for the LSI task.
We curate a large novel dataset for this task, including facts of cases from several major Indian Courts of Law, and statutes from the Indian Penal Code (IPC). 
Modeling the statutes and training documents as a heterogeneous graph, our proposed model **LeSICiN** can learn rich textual and graphical features, and can also tune itself to correlate these features. 
Thereafter, the model can be used to inductively predict links between test documents (new nodes whose graphical features are not available to the model) and statutes (existing nodes). 
Extensive experiments on the dataset show that our model comfortably outperforms several state-of-the-art baselines, by exploiting the graphical structure along with textual features.

# Repo Organization

> - model
  >> - basicmodules.py - Contains basic building blocks -- LSTM and Attention Networks
  >> - submodules.py - Contains Text and Graph Encoders and Matching Network 
  >> - model.py - Contains main module LeSICiN
> - data_helper.py - Helper codes for constructing Dataset and batching logic
> - helper.py - Helper codes for creating vocabularies, label weights, training loop, metrics, etc.
> - run.py - Script for running training and evaluation, and eventual testing

