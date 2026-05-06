# Project Summary

## Recommended repository name

**sms-spam-detection-character-rnn**

This name is hiring-manager-friendly because it clearly communicates:

- SMS spam detection
- NLP classification
- character-level modeling
- recurrent neural networks
- PyTorch engineering implementation

## Alternative names

- `pytorch-sms-spam-rnn`
- `character-level-spam-classifier`
- `sms-spam-gru-classifier`
- `nlp-spam-detection-pytorch`

## GitHub short description

PyTorch character-level GRU model for SMS spam detection with custom batching, class-imbalance handling, and FPR/FNR evaluation.

## LinkedIn project description

Converted an SMS spam detection lab into a modular PyTorch NLP project. The repository includes character-level tokenization, vocabulary creation, custom Dataset/DataLoader logic for variable-length messages, padding via `collate_fn`, a GRU-based classifier, training/validation curves, test-set evaluation, and error analysis using false-positive and false-negative rates.

The model achieved approximately 98.2% test accuracy on the SMS Spam Collection dataset, with additional analysis of operational tradeoffs between missed spam and incorrectly blocked legitimate messages.

## Resume bullets

- Developed a PyTorch character-level GRU classifier for SMS spam detection, implementing vocabulary construction, variable-length batching with `collate_fn`, oversampling for class imbalance, and train/validation/test evaluation.
- Achieved ~98.2% test accuracy and evaluated deployment-relevant error modes using false-positive rate and false-negative rate.
