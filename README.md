# SMS Spam Detection with Character-Level RNN

A PyTorch natural-language-processing project that classifies SMS messages as **spam** or **ham** using a character-level recurrent neural network.

This repository converts a lab notebook into a GitHub-ready engineering project with clean modules, reusable training scripts, custom dataset batching, train/validation/test evaluation, and error analysis using false-positive and false-negative rates.

## Why this project is useful

This project demonstrates practical ML engineering skills relevant for NLP and AI engineering roles:

- Text cleaning and character-level tokenization
- Vocabulary construction with reserved padding token
- Handling variable-length sequences using `collate_fn`
- Custom PyTorch `Dataset` and `DataLoader`
- GRU-based recurrent neural network design
- Class imbalance handling through training-set oversampling
- Training and validation metric tracking
- Test-set evaluation with accuracy, false-positive rate, and false-negative rate
- Inference on custom SMS messages
- Baseline comparison plan for bag-of-words/logistic regression

## Dataset

The project uses the **SMS Spam Collection Dataset** from the UCI Machine Learning Repository.

Expected raw file:

```text
data/SMSSpamCollection
```

Each line should contain:

```text
<label>\t<message>
```

where label is:

- `ham` for non-spam
- `spam` for spam

The lab dataset contained **747 spam** messages and **4,827 ham** messages.

## Repository structure

```text
sms-spam-detection-character-rnn/
├── README.md
├── PROJECT_SUMMARY.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── metrics.py
│   └── utils.py
├── experiments/
│   └── run_experiments.py
├── results/
├── checkpoints/
└── data/
```

## Setup

```bash
git clone <your-repo-url>
cd sms-spam-detection-character-rnn

python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

Place `SMSSpamCollection` inside the `data/` folder.

## Train

```bash
python -m src.train --data-path data/SMSSpamCollection --epochs 15
```

## Evaluate on test set

```bash
python -m src.evaluate --data-path data/SMSSpamCollection --checkpoint checkpoints/best_spam_rnn.pt
```

## Predict one SMS message

```bash
python -m src.predict   --checkpoint checkpoints/best_spam_rnn.pt   --message "machine learning is sooo cool"
```

## Representative lab results

| Metric | Result |
|---|---:|
| Validation accuracy | ~98.8–99.1% |
| Test accuracy | ~98.21% |
| Test false-positive rate | ~1.04% |
| Test false-negative rate | ~6.71% |

The model achieved strong test accuracy, but the false-negative rate was higher than the false-positive rate, meaning some spam messages were missed. In spam detection, false negatives are often especially important because they let spam reach users.

## Resume bullets

- Built a character-level GRU classifier in PyTorch for SMS spam detection, implementing custom text tokenization, variable-length sequence batching with `collate_fn`, class-imbalance handling, training loops, and held-out evaluation.
- Achieved ~98.2% test accuracy on the SMS Spam Collection dataset and analyzed operational risk using false-positive and false-negative rates.
