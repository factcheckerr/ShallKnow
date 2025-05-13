import requests
import json

import os
import re
class LLMQuery:
    def __init__(self, url: str):
        self.api_url = url
        self.api_key = "sk-69cb0447"


    def remove_thinking_part(self, response):
        """
        Removes LLM thinking parts (handles unclosed tags and some variations).
        """
        # Handles <think>...</think>, <reasoning>...</reasoning>, and unclosed tags
        cleaned = re.sub(
            r'(<think>|<reasoning>|<internal>).*?(</think>|</reasoning>|</internal>|$)',
            '',
            response,
            flags=re.DOTALL
        )
        return cleaned.strip()

    def generate_relationship_prompt(self, entity1: str, entity2: str, page_content: str) -> str:
        """
        Generates a prompt to extract relationships between two entities from given content.

        Args:
            entity1: First entity to analyze
            entity2: Second entity to analyze
            page_content: Text content to search for relationships

        Returns:
            Formatted prompt string
        """
        prompt = (
            f"Read the content below and write a natural paragraph using only the information that clearly involves both {entity1} and {entity2}. "
            f"Do not include anything unrelated to these two entities. "
            f"Write only in plain, fluent sentences without formatting, lists, subheadings, or commentary. "
            f"Only include clear and direct or indirect facts that connect or describe {entity1} and {entity2}.\n\n"
            f"Do not generate new text, omit information, or use any external knowledge."
            f"Content:\n{page_content}\n\n"
            f"Do not miss or exclude any information that depicts a connection between {entity1} and {entity2}"
            f"Avoid phrases like 'there is no information about' or 'the only mention is'. "
            f"If the provided text does not mention a direct or indirect connection between {entity1} and {entity2} then output 'No Response'. "

        )
        return prompt.strip()

    def get_response_from_api_call(self, page_content: str, entity1: str, entity2: str, model: str):
        print("ollama starts here")
        url = os.environ.get('OLLAMA_URL', default='http://localhost:11434/api/generate')
        prompt = self.generate_relationship_prompt(entity1, entity2, page_content)
        data = {
            "model": model,
            "keep_alive": -1,
            "prompt": prompt,
            "stream": False,
            "temperature": 0
        }

        try:
            response = requests.post(url, json=data)
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                if line:
                    json_line = line.decode("utf-8")

                    parsed = json.loads(json_line)
                    full_response += parsed.get("response", "")
            cleaned = self.remove_thinking_part(full_response)
            print("LLM response: " +cleaned)
            if full_response:
                full_response = "LLM Output: " + cleaned
            return full_response if full_response else "No response received."

        except requests.exceptions.RequestException as e:
            return f"API error: {e}"

