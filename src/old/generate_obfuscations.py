import google.generativeai as genai

def generate_obfuscations(model, prompt, safety_settings):
    return model.generate_content(prompt, safety_settings = safety_settings)