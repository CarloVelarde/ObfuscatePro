from typing import Union

from .base_model import BaseModel
from src.utility.common_utility import CommonUtility

from google import genai
from google.genai import types


class GeminiModel(BaseModel):

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        super().__init__(api_key, model_name)
        self.client = genai.Client(api_key=self.api_key)

    def _get_prompt_template(
        self,
        code_snippet: str,
        programming_language: str,
        obfuscation_method="ALL METHODS",
    ):
        template = f"""You will be acting as a code obfuscation tool. When I write BEGIN, you will enter this role. All further input from the "User:" will be a request to obfuscate a {programming_language} code snippet using a specified method.
        Here are the steps to follow for each request:
        1. The user will provide the original {programming_language} code snippet inside {{CODE}} tags like this:
        <code>
        {{CODE}}
        </code>

        2. The user will then specify the obfuscation method to use inside {{OBFUSCATION_METHOD}} tags like this:
        <obfuscation_method>
        {{OBFUSCATION_METHOD}}
        </obfuscation_method>

        3. Obfuscate the provided {programming_language} code using the specified method. Some common obfuscation methods include:
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
        """
        return template

    def obfuscate_code(self, code: str, language: str, obfuscation_method: str):

        prompt = self._get_prompt_template(code, language, obfuscation_method)
        response = self.client.models.generate_content(
            model=self.model_name,
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                temperature=0.1,
                system_instruction="You will be acting as a code obfuscation tool. When giving a snippet of code, you will obfuscate it according to the obfuscation method provided.",
            ),
            contents=[prompt],
        )

        formatted_response = CommonUtility.clean_format(response.text)
        return formatted_response

    def embed_code(self, code: Union[str, list], dimensionality: int = 768):
        result = self.client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            content=code,
            config=types.EmbedContentConfig(
                task_type="retrieval_document",
                title="Embedding of single string",
                output_dimensionality=dimensionality,
            ),
        )

        return result.embeddings[0].values
