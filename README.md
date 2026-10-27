# Enron Email Chatbot — Context-Aware Retrieval in the Terminal

A command-line chatbot that answers questions by searching the **Enron Email Corpus** — about half a million real business emails — and keeps track of the conversation so follow-up questions stay on topic. Built as my final project for CS 91R at Swarthmore College.

**Stack:** Python · scikit-learn (TF-IDF) · spaCy · NLTK (VADER) · Wikipedia API

---

## How it works

```
user message ──► spaCy: entities, nouns, question / greeting / farewell
            ──► VADER: sentiment
            ──► context: current topic + last 5 topics + last 10 messages
            ──► TF-IDF + cosine similarity over every email body
            ──► reply with the most relevant email excerpt (and its subject)
```

1. **Index.** On startup, every email is parsed (headers and body), cleaned, and turned into a TF-IDF vector.
2. **Understand.** Each message goes through spaCy for named entities and topic nouns, simple intent detection (question, greeting, farewell), and NLTK's VADER for sentiment.
3. **Remember.** The bot keeps the current topic, the last five topics, recent sentiment, and the last ten messages. When you ask a follow-up, the current topic is added to the search query, so "what happened next?" still searches the right thing.
4. **Retrieve.** The query is compared with every email using cosine similarity, and the bot replies with the first few sentences of the best match, citing its subject line.

If the corpus isn't available, it falls back to a small built-in sample so the bot still runs.

## Commands

| Command | What it does |
|---|---|
| *(any text)* | Chat — get a context-aware answer from the email corpus |
| `wiki <query>` | Look something up on Wikipedia |
| `sentiment <text>` | Score the sentiment of some text |
| `summarize <text>` | Extractive summary using TF-IDF sentence scoring |
| `clear` | Reset conversation context |
| `debug` | Show dataset status and a sample email |
| `help` | List commands |
| `exit` | Quit |

## Running it

```bash
git clone https://github.com/num-err/enro.git
cd enro/Enro
pip install nltk spacy wikipedia-api numpy scikit-learn python-dateutil colorama
python -m spacy download en_core_web_sm
python chatbot.py
```

**Dataset:** download the [Enron Email Dataset](https://www.cs.cmu.edu/~enron/) and point `enron_path` in `chatbot.py` to the folder that contains `maildir/`. Without it, the bot uses its built-in sample emails.

## Files

```
Enro/
  chatbot.py        the chatbot
  writeup.md        project write-up: design, libraries, and results
  requirements.txt  dependencies
```

## Next steps

- Sentence embeddings for meaning-based retrieval
- LLM summarization of retrieved emails into direct answers
- A trained intent classifier
