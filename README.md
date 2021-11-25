# AlephBERT

## overview

A large Pre-trained language model for Modern Hebrew

Based on BERT-base training, 12 hidden layers, with 52K vocab size.

Trained on 95M sentences from OSCAR+Wikipedia+Tweeter data, 10 epochs.

## Evaluation

We evaluated AlephBERT for the following prediction tasks:

- Morphological Segmentation
- Part of Speech Tagging
- Morphological Features
- Named Entity Recognition
- Sentiment Analysis

On four different benchmarks:

- The SPMRL Treebank (for: Segmentation, POS, Feats, NER)
- The Universal Dependency Treebanks  (for: Segmentation, POS, Feats, NER)  
- The Hebrew Facebook Corpus (for: Sentiment Analysis)

## Citation

@misc{alephBert2021,

      title={AlephBERT: a Pre-trained Language Model to Start Off your Hebrew NLP Application}, 
      
      author={Amit Seker, Elron Bandel, Dan Bareket, Idan Brusilovsky, Shaked Refael Greenfeld, Reut Tsarfaty},
      
      year={2021}
      
}

## Contributors:

The ONLP Lab at Bar Ilan University

PI: Prof. Reut Tsarfaty

Contributor: Amit Seker, Elron Bandel, Dan Bareket, Idan Brusilovsky, Shaked Refael Greenfeld

Advisors: Dr. Roee Aharoni, Prof. Yoav Goldberg


## Credits

- The Hebrew Treebank: http://www.cs.technion.ac.il/~itai/publications/NLP/TreeBank.pdf
- The SPMRL Shared Task data: https://www.aclweb.org/anthology/W13-4917/
- The Universal Dependencies Treebank: https://www.aclweb.org/anthology/W18-6016
- The Named Entity Recognition Annotations: https://arxiv.org/abs/2007.15620
- The Hebrew Sentiment Analysis Corpus: https://github.com/omilab/Neural-Sentiment-Analyzer-for-Modern-Hebrew


