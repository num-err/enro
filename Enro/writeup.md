# CS91R Final Project Writeup: Context-Aware Conversational Chatbot

# Project Overview
This project was developed by Numer Ahmed  as a final project for CS91R. I created an intelligent chatbot that combines multiple natural language processing techniques to provide context-aware responses and insights. The chatbot leverages the Enron Email Corpus to generate relevant responses while maintaining conversation context.

# External Tools and Libraries Used

## Core NLP Libraries
- NLTK: Used for sentiment analysis, tokenization, and text processing
- spaCy: Implemented for advanced NLP tasks like entity recognition and intent analysis
- Wikipedia API: Integrated for retrieving additional information when needed

# Data Processing and ML
- NumPy: Used for numerical operations and array manipulations
- scikit-learn: Implemented for text vectorization and similarity calculations
- TfidfVectorizer: Used for converting text into numerical features

# Dataset
- Enron Email Corpus: Primary dataset used for training and response generation
  - Contains thousands of real business emails
  - Provides rich context for business-related queries
  - Helps in generating relevant and contextual responses

# Implementation Details

# Key Features

1. Context-Aware Conversations
   - Maintains conversation history
   - Tracks user interests and topics
   - Provides personalized responses

2. Advanced NLP Capabilities
   - Sentiment analysis of user input
   - Intent recognition for better understanding
   - Entity extraction for topic identification
   - Text summarization for concise responses

3. Interactive Commands
   - `help`: Shows available commands
   - `wiki <query>`: Searches Wikipedia
   - `sentiment <text>`: Analyzes text sentiment
   - `summarize <text>`: Summarizes given text
   - `clear`: Resets conversation context
   - `debug`: Shows dataset status
   - `exit`: Closes the chatbot

### Technical Implementation

1. Package Management
   - Graceful handling of missing dependencies
   - Automatic download of required NLTK data
   - Fallback mechanisms for unavailable features

2. Data Processing
   - Efficient email parsing and cleaning
   - Smart text preprocessing
   - Context management system

3. Response Generation**
   - Semantic similarity search
   - Context-aware response selection
   - Fallback mechanisms for unknown queries

## Setup Instructions

1. Environment Setup
   ```bash
   # Create and activate virtual environment
   python3 -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   .\venv\Scripts\activate  # On Windows
   ```

2. Install Dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. Dataset Setup**
   - The chatbot will automatically look for the Enron dataset in default locations
   - Alternatively, specify the path using the `--enron-path` argument

## Usage Guide

1. Starting the Chatbot
   ```bash
   python chatbot.py --enron-path /path/to/enron/dataset
   ```
   ```
   type "debug" to see the details of the enron data set
   ```

2. Basic Interaction
   - Type your message and press Enter
   - The chatbot will respond based on context and available information
   - Use commands for specific features (e.g., `help`, `wiki`, `sentiment`)

3. Example Interactions
   ```
   > help
   [Shows available commands and features]

   > wiki artificial intelligence
   [Displays Wikipedia information about AI]

   > sentiment I'm really happy with this project
   [Shows sentiment analysis results]
   ```

## Challenges and Solutions

1.Data Processing and parsing
   - Challenge: Handling large email dataset efficiently
   - Solution: Implemented smart sampling and caching

2. Context Management
   - Challenge: Maintaining relevant conversation context
   - Solution: Developed a context tracking system with history

3. Response Generation
   - Challenge: Generating relevant and coherent responses
   - Solution: Combined multiple NLP techniques with fallback mechanisms

# Future Improvements

1. Enhanced Features
   - Multi-language support
   - Web interface
   - Real-time learning capabilities

2. Technical Enhancements
   - Improved response generation
   - Better context management
   - Enhanced error handling



On all the above note , We would like to thank:
- The CS91R teaching staff for their guidance
- The Enron Email Corpus for providing valuable data
- The open-source community for the excellent NLP libraries

## Conclusion

This project successfully demonstrates the practical application of various NLP techniques I learned throughout the semester. The chatbot provides a user-friendly interface for exploring these concepts while maintaining context and generating relevant responses. I hope this tool serves as a useful example of how different NLP techniques can be combined to create an intelligent conversational system. 