# Obfuscate Pro

#### Inspired by Work Done During NSF REU program

#### Maintainers: Carlo Velarde

## About

Obfuscator Pro is a Python-based utility designed to simplify the process of code obfuscation using AI models, specifically the Gemini and GPT models. This tool acts as a wrapper around AI models, making it easier for users to obfuscate code, generate embeddings, find similarities, and build tables. It optimizes model parameters, prompts, and context to create high quality obfuscations. It is ideal for developers and researchers working on code obfuscation, security, and related areas.

## Setup

#### Clone the Repo

#### Create Virtual Enviroment

`python -m venv venv`

#### Activating virtual enviroment

- For **mac**
  `source venv/bin/activate`

- For **windows**
  `.\venv\Scripts\activate`

_When the virtual enviroment is installed and activated please install the requirements with the code below._

#### Installing required code

`pip install -r requirements.txt`

### Setup Program

This utility works as objects. Each model (GPT, GEMINI) has its own class. To set up you can do

```python
from models.gemini_model import GeminiModel

model = GeminiModel(api_key = GOOGLE_API_KEY, model_name = "gemini-2.0-flash")

obfuscated_code = model.obfuscate_code(
    code=code_snippet, language="Java", obfuscation_method="Naming and Dead Code"
)
```
