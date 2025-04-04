import os
import requests


def prompt_ollama(prompt, model="llama3.1:latest", api_url="http://10.167.31.201:11434/api/generate"):
    """
    Queries the Ollama API.
    
    Args:
        prompt (str): The prompt to send to the model.
        model (str): The name of the model to use.
        api_url (str): The endpoint for the Ollama API.
    
    Returns:
        string: The response from the model.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        output = result.get("response", "")
        return output
    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama API: {e}")
        return []