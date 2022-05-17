# Viterbi Algorithm for Parts of Speech Tagging

This project is an implementation of the Viterbi algorithm for Parts of Speech tagging of any language supported by the Universal Dependencies Treebanks.

### Algorithms
- Greedy algorithm
- Viterbi algorithm
- Forward-Backward algorithm

### Dependencies
- nltk
- conllu
- matplotlib
- seaborn
- sklearn
- numpy

### Execution
Download the required treebanks from Universal Dependencies (english ewt, japanese gsd, spanish gsd, arabic padt, dutch alpino) from the Universal Dependencies site (https://universaldependencies.org/#download).
```
python main.py
```
The script above will execute all three algorithms for all five language treebanks and report results on console and save confusion matrix figures in the figs/ subfolder.

### Test Accuracy Results for English POS Tagging
Greedy: 87.5%
Viterbi: 89.5
Forward-Backward: 85.0%

### Known Bugs
- Forward Backward algorithm contains a bug as it should have a better accuracy than Viterbi.