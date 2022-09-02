# Healthcare Chatbot

This is a Python-based project for dealing with human symptoms and predicting their possible outcomes.

#### Project Goal
The primary goal of this project is to forecast the disease so that patients can get the desired output according to their primary symptoms.

## Team Members
1. Durgesh Maurya
2. Dimpal
3. Dev shran Yadav 
 
## Technology used
We used **[TKinter](https://docs.python.org/3/library/tkinter.html)** to create a desktop-based application and **[Spacy](https://spacy.io/)** for NLP-based processes like ***text sentence tokenization and lemmatization***, and we used a **[Huggingface](https://huggingface.co/)** pretrained model to extrat disease names from a given sentence ***( or ner processing)***.

#### Huggingface
Downloading pre-trained model from [Huggingface Model](https://huggingface.co/raynardj/ner-disease-ncbi-bionlp-bc5cdr-pubmed)
```python

from transformers import pipeline
PRETRAINED = "raynardj/ner-disease-ncbi-bionlp-bc5cdr-pubmed"
ners = pipeline(task="ner",model=PRETRAINED, tokenizer=PRETRAINED)

```

#### Spacy

Download [spacy](https://spacy.io/usage) For window, Linux, MacOS
```bash
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
```


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.txt file package.

```bash
pip install -r requirements.txt 
```
## Output Image
#### First opening window
![first window](https://github.com/Durgesh63/HealthCare_ChatBot/blob/master/firstwindow.png?raw=true)

#### Main Opening Window
![Main Window](https://github.com/Durgesh63/HealthCare_ChatBot/blob/master/main_window.png?raw=true)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
