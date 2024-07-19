# Obfuscate Pro
####  Developed during NSF REU program
#### Maintainers: Carlo Velarde

## About

Obfuscator Pro is a Python-based utility designed to simplify the process of code obfuscation using AI models, specifically the Gemini model. This tool acts as a wrapper around AI models, making it easier for users to obfuscate code, generate embeddings, find similarities, and build tables. It optimizes model parameters, prompts, and context to create high quality obfuscations. It is ideal for developers and researchers working on code obfuscation, security, and related areas.

## Setup
#### Clone the Repo

#### Create Virtual Enviroment

`python -m venv venv`

#### Activating virtual enviroment

* For __mac__
   `source venv/bin/activate`

* For __windows__
   `.\venv\Scripts\activate`

*When the virtual enviroment is installed and activated please install the requirements with the code below.*

#### Requirements code

`pip install -r requirements.txt`


### Run Program

The entry point of the program is obfuscator.py. Set up the model by `obfuscator = Obfuscator(GOOGLE_API_KEY, model = "GEMINI")`





