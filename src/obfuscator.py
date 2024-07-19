import time
import os

import google.generativeai as genai
import pandas as pd
from dotenv import load_dotenv
from generate_obfuscations import generate_obfuscations
from formatter import clean_format
from table import Table
from typing import Union
from similarity import calculate_cosine_similarity, calculate_euclidean_distance

class Obfuscator:
    def __init__(self, apiKey:str, model = "GEMINI"):
        if model.upper() == "GEMINI":
            genai.configure(api_key = apiKey)
            self.model = genai.GenerativeModel('gemini-1.5-pro')
            self.safety_settings = [
                {
                    "category": "HARM_CATEGORY_DANGEROUS",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]
    

    def set_safety_level(self, level=1):

        if level < 1 or level > 3:
            raise ValueError(f"{level} is out of the range 1-3")
        
        elif level == 1:
            self.safety_settings = [
                {
                    "category": "HARM_CATEGORY_DANGEROUS",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]
        elif level == 2:
            self.safety_settings = [
                {
                    "category": "HARM_CATEGORY_DANGEROUS",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_ONLY_HIGH",
                },
            ]
            
        elif level == 3:
            self.safety_settings = [
                {
                    "category": "HARM_CATEGORY_DANGEROUS",
                    "threshold": "BLOCK_LOW_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_LOW_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_LOW_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_LOW_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_LOW_AND_ABOVE",
                },
            ]
    

    # Helper method for generating obfuscations
    def get_prompt_template(self, code_snippet: str, obfuscation_method = "ALL METHODS", language = ""):
        template = f'''You will be acting as a code obfuscation tool. When I write BEGIN, you will enter this role. All further input from the "User:" will be a request to obfuscate a {language} code snippet using a specified method.
        Here are the steps to follow for each request:
        1. The user will provide the original {language} code snippet inside {{CODE}} tags like this:
        <code>
        {{CODE}}
        </code>

        2. The user will then specify the obfuscation method to use inside {{OBFUSCATION_METHOD}} tags like this:
        <obfuscation_method>
        {{OBFUSCATION_METHOD}}
        </obfuscation_method>

        3. Obfuscate the provided {language} code using the specified method. Some common obfuscation methods include:
        - Renaming variables and functions
        - Inserting dead code or dummy statements
        - Splitting strings
        - Encoding strings or numbers
        - Flattening the control flow
        - Using complex expressions or formulas
        Only use the obfuscation method that the user specifies.

        4. Ignore spacing and line breaks. Only include code. To separate lines use the ";" syntax. 

        5. After obfuscating the code, provide the obfuscated version as a STRING. Ensure that the only output you return is the code represented as a string.

        6. Ensure that the obfuscated code, when executed, produces the same output as the original code. The goal is to make the code harder to understand while preserving its functionality.

        Remember:
        - Only obfuscate the code using the method specified by the user.
        - The obfuscated code should be functionally equivalent to the original code.
        - Do not explain your obfuscation process or include any additional commentary.
        - Provide only the code as a string.

        BEGIN

        <code>
        {code_snippet}
        </code>

        <obfuscation_method>
        {obfuscation_method}
        </obfuscation_method>
        '''
        return template
    

    def obfuscate(self, prompt, safety_settings = None):
        if safety_settings is None:
            safety_settings = self.safety_settings

        response = generate_obfuscations(model = self.model, prompt = prompt, safety_settings = safety_settings)
        
        obfuscated_code = response.text
        # Clean formatting 
        cleaned_obfuscated_code = clean_format(obfuscated_code)


        return cleaned_obfuscated_code
    
    def embed(self, code: Union[str, list], dimensionality=768):
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=code,
            task_type="retrieval_document",
            title="Embedding of single string",
            output_dimensionality = dimensionality
            )

        return result['embedding']
    
    def similarity(self, emb1: list, emb2: list, algo_type = None, round_to = None):
        similarity = 0

        if algo_type is None:
            algo_type = "COSINE_SIMILARITY"

        if algo_type == "COSINE_SIMILARITY":
            similarity = calculate_cosine_similarity(emb1, emb2)
            if round_to is not None:
                similarity = round(similarity, round_to)
        
        elif algo_type == "EUCLIDEAN_DISTANCE":
            similarity = calculate_euclidean_distance(emb1, emb2)
            if round_to is not None:
                similarity = round(similarity, round_to)

        return similarity
    

    def table(self):
        return Table(pd.DataFrame())
    
    
    
    


if __name__ == "__main__":


    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    obfuscator = Obfuscator(GOOGLE_API_KEY, model = "GEMINI")

    obfuscator.set_safety_level(1)

    # prompt = obfuscator.get_prompt_template(code_snippet = "let name = 'Carlo'; console.log(name)", language = "JavaScript", obfuscation_method = "naming")


    # obfuscation = obfuscator.obfuscate(prompt = prompt, safety_settings = obfuscator.safety_settings)

    # embedding1 = obfuscator.embed(obfuscation)
    # time.sleep(1)
    # embedding2 = obfuscator.embed("let name = 'james'; console.log(name)")

    # print(f'Cosine Similarity: {obfuscator.similarity(embedding1, embedding2, algo_type = "COSINE_SIMILARITY", round_to = 4)}')

    # print(f'EUCLIDEAN_DISTANCE: {obfuscator.similarity(embedding1, embedding2, algo_type = "EUCLIDEAN_DISTANCE", round_to = 4)}')

    table = obfuscator.table()
    table.add_column("Embeddings")
    print(table.table.head())
    table.save_csv("./data/temp.csv")
    





    




    

