# AI Chatbot for Student Advisory Using NLP

## Project Overview

This project aims to develop an AI chatbot for student advisory services using Natural Language Processing (NLP). The chatbot is designed to assist students with inquiries related to their academic and administrative needs. The project leverages Python, Django, BERT model for NLP, and several frontend technologies to create a responsive and interactive chatbot system.

## Features

- **Student Advisory Services**: Provides automated responses to student inquiries.
- **Authentication System**: Allows users to sign up and log in using email, Google, or Microsoft accounts.
- **Chat History**: Users can view their previous chat history upon logging in.
- **Theme Mode**: Offers both light and dark modes for better user experience.
- **Error Handling**: Custom error pages for better user guidance.

## Front-End

- **HTML/CSS/JavaScript**: The basic building blocks of the frontend.
- **Bootstrap**: Used for responsive design and styling.
- **Wireframes and Mockups**: Created using Adobe XD to design the user interface.
- **Font and Colors**: 
  - Font: Montserrat
  - Colors: 
    - Primary: #0093FA
    - Dark Mode: Black
    - Light Mode: White

### Front-End Features

- **Error 404 Page**: Provides a user-friendly message when a page is not found.
- **Success Page**: Displays a confirmation message upon successful actions (e.g., form submission).
- **Welcome Page**: Greets users upon successful login.
- **Hide Password Toggle**: Allows users to toggle the visibility of their password input.

## Back-End

- **Django**: Used as the web framework for the backend.
- **Python**: The primary programming language for backend logic and NLP model integration.
- **SQL**: Database management for storing user data and chat history.

### Backend Components

- **urls.py**: Defines URL patterns for the application.
- **views.py**: Contains the logic for handling requests and rendering responses.
- **models.py**: Defines the database models.
- **forms.py**: Manages user input forms.
- **admin.py**: Configures the admin interface.
- **asgi.py and wsgi.py**: ASGI and WSGI configurations for deployment.
- **init.py**: Indicates that the directory should be treated as a package.
- **manage.py**: Command-line utility for administrative tasks.
- **pycache**: Compiled Python files.

## NLP Model Integration

- **BERT Model**: Used for processing and understanding natural language queries from students.
- **Google Colab**: Environment used for training the BERT model.
- **Transformers Library**: Used for model implementation.

## Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/student-advisory-chatbot.git
   cd student-advisory-chatbot
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Migrations**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

4. **Run the Server**
   ```bash
   python manage.py runserver
   ```

5. **Access the Application**
   Open your web browser and navigate to `http://127.0.0.1:8000/`

## Training the NLP Model

1. **Load the Dataset**
   Ensure your dataset is available in the correct format for training.
   
2. **Train the Model**
   Use the provided `train_model.py` script to train the BERT model on your dataset.
   
   ```python
   from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments

   # Initialize the tokenizer and model
   tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
   model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

   # Training arguments
   training_args = TrainingArguments(
       output_dir='./results',
       num_train_epochs=3,
       per_device_train_batch_size=8,
       per_device_eval_batch_size=8,
       warmup_steps=500,
       weight_decay=0.01,
       logging_dir='./logs',
   )

   # Trainer
   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_dataset,
       eval_dataset=eval_dataset
   )

   # Train the model
   trainer.train()
   ```

# Inline Comments within the Code
```python

# AI Chatbot for Student Advisory Using NLP

"""
This file contains the code for the AI chatbot project aimed at providing student advisory services.
"""

# Importing necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sklearn.model_selection as model_selection
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import transformers
from transformers import AutoModel, BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import torch.nn.functional as F
import nltk
import string
import random

# Rest of the code is in the Repository
```

## Contributions

Contributions are welcome! Please fork this repository and submit pull requests for any features or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](  http://www.apache.org/licenses/) file for more information.
