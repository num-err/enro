#!/usr/bin/env python3

import sys
import os
import re
from datetime import datetime
from collections import deque
import random

# Import required packages with graceful fallbacks
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    for download in ['punkt', 'stopwords', 'vader_lexicon']:
        try:
            nltk.data.find(f'tokenizers/{download}' if download == 'punkt' else 
                         f'corpora/{download}' if download == 'stopwords' else 
                         f'sentiment/{download}')
        except LookupError:
            nltk.download(download)
except ImportError:
    print("Warning: NLTK not available. Some features will be limited.")
    nltk = None

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except ImportError:
    print("Warning: spaCy not available. Some features will be limited.")
    nlp = None

try:
    import wikipediaapi
    wiki = wikipediaapi.Wikipedia(user_agent='cs91r-chatbot/1.0 (nahmed3@swarthmore.edu)', language='en')
except ImportError:
    print("Warning: Wikipedia API not available. Wiki search will be limited.")
    wiki = None

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Warning: scikit-learn not available. Summarization will be limited.")
    np = None

class Chatbot:
    def __init__(self):
        # Initialize NLP components
        self.sia = SentimentIntensityAnalyzer() if nltk else None
        self.stopwords = set(stopwords.words('english')) if nltk else set()
        
        # Initialize context management
        self.conversation_history = deque(maxlen=10)
        self.user_context = {
            'name': None, 'interests': set(), 'last_topics': deque(maxlen=5),
            'sentiment_history': deque(maxlen=5), 'current_topic': None
        }
        
        # Initialize email data structures
        self.enron_data = []
        self.email_vectors = None
        self.vectorizer = None
        
        print("Initializing chatbot...")
        self.load_enron_dataset()
        
        self.commands = {
            'help': self.show_help, 'wiki': self.wiki_search,
            'sentiment': self.analyze_sentiment, 'summarize': self.summarize_text,
            'clear': self.clear_context, 'exit': self.exit_chatbot,
            'debug': self.debug_enron
        }

    def load_enron_dataset(self):
        """Load and prepare Enron dataset."""
        self.enron_data = self.load_enron_data()
        if not self.enron_data:
            print("No real Enron data found, using sample data...")
            self.enron_data = self.create_sample_data()
        
        email_texts = [email['body'].lower() if isinstance(email, dict) else email.lower() 
                      for email in self.enron_data]
        
        if email_texts and np:
            try:
                self.vectorizer = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.95)
                self.email_vectors = self.vectorizer.fit_transform(email_texts)
                print(f"Vectorized {len(email_texts)} emails")
            except Exception as e:
                print(f"Vectorization failed: {str(e)}")
                self.vectorizer = None
                self.email_vectors = None

    def load_enron_data(self):
        """Load and preprocess Enron email dataset."""
        try:
            enron_path = '/data/cs91r-s25/corpora/enron'
            if not os.path.exists(enron_path):
                print(f"[DEBUG] Enron dataset not found at: {enron_path}")
                return self.create_sample_data()
            
            print(f"Loading Enron dataset from: {enron_path}")
            emails = []
            email_count = skipped_count = 0
            header_pattern = re.compile(r'([\w-]+):\s*(.*)')
            
            all_files = [os.path.join(root, f) for root, _, files in os.walk(os.path.join(enron_path, 'maildir')) for f in files]
            print(f"Found {len(all_files)} email files")
            
            sample_size = min(10000, len(all_files))
            sampled_files = random.sample(all_files, sample_size) if len(all_files) > sample_size else all_files
            
            for file_path in sampled_files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        headers, body = self._parse_email(content, header_pattern)
                        body = self._clean_text(body)
                        
                        if len(body) < 50:
                            skipped_count += 1
                            continue
                        
                        folder_path = os.path.dirname(file_path).replace(enron_path, '').strip('/')
                        metadata = {
                            'folder': folder_path, 'subject': headers.get('subject', ''),
                            'from': headers.get('from', ''), 'to': headers.get('to', ''),
                            'date': headers.get('date', '')
                        }
                        
                        emails.append({'body': body, 'metadata': metadata})
                        email_count += 1
                        
                        if email_count % 1000 == 0:
                            print(f"Loaded {email_count} emails...")
                except Exception as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue
            
            print(f"Successfully loaded {len(emails)} emails")
            print(f"Skipped {skipped_count} emails")
            
            if emails:
                self._print_sample_email(emails[0])
            
            return emails
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return self.create_sample_data()

    def _parse_email(self, content, header_pattern):
        """Parse email content into headers and body."""
        parts = content.split("\n\n", 1)
        headers = {}
        if len(parts) == 2:
            header_section, body = parts
            for line in header_section.split('\n'):
                match = header_pattern.match(line)
                if match:
                    key, value = match.groups()
                    headers[key.lower()] = value
        else:
            body = content
        return headers, body

    def _clean_text(self, text):
        """Clean and normalize text."""
        text = re.sub(r'^\s*>.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'_{2,}|-{2,}|={2,}', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _print_sample_email(self, email):
        """Print sample email for debugging."""
        print("\nSample email:")
        print(f"Subject: {email['metadata']['subject']}")
        print(f"From: {email['metadata']['from']}")
        print(f"Body preview: {email['body'][:200]}...")

    def create_sample_data(self):
        """Create sample email data for testing."""
        print("Creating sample data...")
        return [
            {
                'body': "Regarding natural gas trading, our position in Q3 showed a 15% increase over projections. The team expects continued growth in Q4 despite market volatility.",
                'metadata': {
                    'subject': 'Q3 Gas Trading Position',
                    'from': 'john.doe@enron.com',
                    'to': 'trading.team@enron.com',
                    'date': 'Mon, 15 Oct 2001 09:32:17 -0700'
                }
            },
            {
                'body': "Energy prices have continued to climb through September. Our analytics team predicts a 7-10% increase in wholesale electricity prices by year end. Recommend adjusting our hedging strategy accordingly.",
                'metadata': {
                    'subject': 'Energy Price Forecast',
                    'from': 'sarah.smith@enron.com',
                    'to': 'exec.committee@enron.com',
                    'date': 'Wed, 3 Oct 2001 14:25:36 -0700'
                }
            },
            {
                'body': "The oil futures we discussed last week have already moved up 3 points. If you want to take that position we talked about, I'd suggest doing it before the end of the week.",
                'metadata': {
                    'subject': 'Oil Futures Position',
                    'from': 'trader1@enron.com',
                    'to': 'trader2@enron.com',
                    'date': 'Tue, 18 Sep 2001 08:14:22 -0700'
                }
            }
        ]

    def find_relevant_emails(self, query, top_k=3):
        """Find most relevant emails based on semantic similarity."""
        # explicit checks avoid ambiguous truth‐value errors
        if self.vectorizer is None or self.email_vectors is None or not self.enron_data:
            print("Semantic search unavailable: missing components")
            return []

        query_vec = self.vectorizer.transform([query.lower().strip()])
        sims = cosine_similarity(query_vec, self.email_vectors).flatten()
        top_idxs = sims.argsort()[-top_k:][::-1]

        results = []
        for i in top_idxs:
            score = float(sims[i])
            if score >= 0.05:
                results.append({
                    'email': self.enron_data[i],
                    'similarity': score,
                    'preview': self.enron_data[i]['body'][:300]
                })
        return results

    def debug_enron(self, args):
        """Show Enron dataset debugging information."""
        print("\n--- ENRON DATASET DEBUG INFO ---")
        print(f"NumPy available: {np is not None}")
        print(f"Dataset loaded: {len(self.enron_data) > 0}")
        print(f"Number of emails: {len(self.enron_data)}")
        print(f"Vectorizer initialized: {self.vectorizer is not None}")
        
        if hasattr(self, 'email_vectors') and self.email_vectors is not None:
            print(f"Email vectors shape: {self.email_vectors.shape}")
        
        if self.enron_data:
            print("\nSample email:")
            sample = self.enron_data[0]
            if isinstance(sample, dict) and 'metadata' in sample:
                print(f"Subject: {sample['metadata'].get('subject', 'N/A')}")
                print(f"From: {sample['metadata'].get('from', 'N/A')}")
                print(f"Body preview: {sample['body'][:150]}...")
        
        print("\nTesting search...")
        for q in ["energy", "gas", "trading", "business"]:
            matches = self.find_relevant_emails(q, top_k=1)
            if matches:
                print(f"Query '{q}' → score {matches[0]['similarity']:.3f}")
            else:
                print(f"Query '{q}' → no matches")

    def analyze_intent(self, text):
        """Analyze user intent using available NLP techniques."""
        if not nlp:
            return {'intents': {'greeting': 'hi' in text.lower(), 'farewell': 'bye' in text.lower()}}
        
        doc = nlp(text.lower())
        return {
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'intents': {
                'question': any(token.dep_ == 'ROOT' and token.tag_ == 'WP' for token in doc),
                'greeting': any(token.text in ['hi', 'hello', 'hey'] for token in doc),
                'farewell': any(token.text in ['bye', 'goodbye', 'see you'] for token in doc),
                'topic': [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
            },
            'sentiment': self.sia.polarity_scores(text) if self.sia else None
        }

    def show_help(self, *args):
        """Show available commands."""
        print("""
Available commands:
  help                    - Show this help message
  wiki <query>           - Search Wikipedia for information
  sentiment <text>       - Analyze sentiment of given text
  summarize <text>       - Summarize the given text
  clear                  - Clear conversation context
  debug                  - Show Enron dataset debugging information
  exit                   - Exit the chatbot

Features:
  - Maintains conversation history
  - Recognizes user intents
  - Provides personalized responses
  - Uses Enron email dataset for relevant information
        """)

    def wiki_search(self, query):
        """Search Wikipedia for information."""
        if not wiki or not query:
            print("Wikipedia search is not available." if not wiki else "Please provide a search query.")
            return
        
        page = wiki.page(query)
        if page.exists():
            print(f"\nTitle: {page.title}\n\nSummary:\n{page.summary[:500]}...")
        else:
            print(f"No Wikipedia page found for '{query}'")

    def analyze_sentiment(self, text):
        """Analyze sentiment of given text."""
        if not self.sia or not text:
            print("Sentiment analysis is not available." if not self.sia else "Please provide text to analyze.")
            return
        
        scores = self.sia.polarity_scores(text)
        print("\nSentiment Analysis Results:")
        print(f"Positive: {scores['pos']:.2f}")
        print(f"Neutral:  {scores['neu']:.2f}")
        print(f"Negative: {scores['neg']:.2f}")
        print(f"Compound: {scores['compound']:.2f}")

    def summarize_text(self, text):
        """Summarize the given text."""
        if not all([np, nlp]) or not text:
            print("Text summarization is not available." if not all([np, nlp]) else "Please provide text to summarize.")
            return
        
        doc = nlp(text)
        sentences = list(doc.sents)
        
        if len(sentences) <= 3:
            print("Text is already short enough.")
            return
        
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([sent.text for sent in sentences])
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        top_indices = sentence_scores.argsort()[-3:][::-1]
        summary = [sentences[i].text for i in sorted(top_indices)]
        
        print("\nSummary:")
        print(" ".join(summary))

    def clear_context(self, *args):
        """Clear conversation context."""
        self.conversation_history.clear()
        self.user_context = {
            'name': None, 'interests': set(), 'last_topics': deque(maxlen=5),
            'sentiment_history': deque(maxlen=5), 'current_topic': None
        }
        print("Conversation context cleared.")

    def exit_chatbot(self, *args):
        """Exit the chatbot."""
        print("Goodbye!")
        sys.exit(0)

    def process_command(self, user_input):
        """Process user input and execute corresponding command."""
        analysis = self.analyze_intent(user_input)
        self.conversation_history.append({
            'input': user_input,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        })
        
        parts = user_input.strip().split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command in self.commands:
            self.commands[command](args)
        else:
            response = self.generate_response(user_input, analysis)
            if response:
                print(f"\nChatbot: {response}")

    def generate_response(self, user_input, analysis):
        """Generate a context-aware response."""
        if analysis['intents']['greeting']:
            return f"Hello {self.user_context['name']}!" if self.user_context['name'] else "Hello! How can I help you today?"
        
        if analysis['intents']['farewell']:
            return "Goodbye! Have a great day!"
        
        # Update context
        topics = analysis['intents'].get('topic', [])
        if topics:
            self.user_context['current_topic'] = topics[0]
            self.user_context['last_topics'].append(topics[0])
        
        if self.sia:
            sentiment = self.sia.polarity_scores(user_input)
            self.user_context['sentiment_history'].append(sentiment['compound'])
        
        # Search for relevant information
        search_query = f"{self.user_context['current_topic']} {user_input}" if self.user_context['current_topic'] else user_input
        relevant_emails = self.find_relevant_emails(search_query)
        
        if relevant_emails:
            top_result = relevant_emails[0]
            email = top_result['email']
            sentences = email['body'].split('.')
            response = '. '.join(sentences[:3]) + '.' if len(sentences) > 2 else email['body'][:300] + '...'
            prefix = f"I found this from an email about '{email['metadata'].get('subject')}': " if email['metadata'].get('subject') else "I found this relevant information: "
            return prefix + response
        
        # Fallback response
        if self.user_context['current_topic']:
            return f"I couldn't find specific information about {self.user_context['current_topic']}. Could you provide more details?"
        return "I don't have specific information about that. Could you tell me more about what you're interested in?"

    def run(self):
        """Run the chatbot main loop."""
        print("Welcome to the NLP Chatbot!")
        print("Type 'help' to see available commands.")
        print("Type 'debug' to see Enron dataset status.")
        
        while True:
            try:
                user_input = input("\n> ")
                if user_input.strip():
                    self.process_command(user_input)
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot.run()