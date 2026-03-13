import google.generativeai as genai
import openai
from openai import APIConnectionError
import json
import logging
from typing import Optional
import re
import requests  # NEW: Import for Hugging Face API calls


def _call_gemini_llm(api_key: str, model_name: str, prompt: str) -> tuple[str, dict]:
    """
    Calls the Google Gemini LLM with the given API key, model name, and prompt.
    Expects the LLM to return a JSON string with a 'suggestions' key.
    """
    if not api_key:
        raise ValueError("Gemini API Key is not configured.")
    if not model_name:
        raise ValueError("Gemini Model Name is not configured.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    try:
        response = model.generate_content(prompt)
        raw_response_text = response.text  # Keep the original raw text

        # Find the JSON block using a regular expression
        json_match = re.search(
            r"```(json)?\s*({.*})\s*```", raw_response_text, re.DOTALL
        )

        if json_match:
            # Extract the JSON string from the regex match
            json_str = json_match.group(2)
        else:
            # If no markdown fence is found, assume the whole response is the JSON string
            json_str = raw_response_text.strip()

        # Parse the extracted JSON string
        parsed_response = json.loads(json_str)

        # Return the ORIGINAL raw text and the parsed dictionary
        return raw_response_text, parsed_response
    except Exception as e:
        logging.error(f"Error calling Gemini LLM: {e}")
        raise RuntimeError(f"Failed to get response from Gemini LLM: {e}")


def _call_huggingface_llm(
    api_key: Optional[str], base_endpoint: str, model_id: str, prompt: str
) -> tuple[str, dict]:
    """
    Calls a Hugging Face Inference API endpoint with the given API key (optional), base endpoint, model ID, and prompt.
    Expects the LLM to return a JSON string.
    """
    if not base_endpoint:
        raise ValueError("Hugging Face Base Endpoint is not configured.")
    if not model_id:
        raise ValueError("Hugging Face Model ID is not configured.")

    # Construct the full model endpoint URL
    model_endpoint = f"{base_endpoint.rstrip('/')}/{model_id.lstrip('/')}"
    logging.info(f"[LLM] Hugging Face Model Endpoint: {model_endpoint}")

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Hugging Face Inference API expects a list of inputs,
    # and we're expecting JSON output from the model.
    # The `return_full_text=False` is important for instruction-tuned models
    # to not return the input prompt along with the generated text.
    payload = {
        "inputs": prompt,
        "parameters": {
            "return_full_text": False,
            "max_new_tokens": 2048,  # Default max tokens, can be made configurable
            "temperature": 0.7,  # Default temperature, can be made configurable
            # "top_p": 0.9           # Can be added if needed
        },
    }

    try:
        response = requests.post(model_endpoint, headers=headers, json=payload)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        raw_response_data = response.json()

        if (
            not raw_response_data
            or not isinstance(raw_response_data, list)
            or "generated_text" not in raw_response_data[0]
        ):
            raise ValueError(
                f"Unexpected response format from Hugging Face API: {raw_response_data}"
            )

        raw_text = raw_response_data[0]["generated_text"]

        # Try to find and parse a JSON block within the generated text
        json_match = re.search(r"```(json)?\s*({.*})\s*```", raw_text, re.DOTALL)

        if json_match:
            json_str = json_match.group(2)
        else:
            # If no markdown fence is found, assume the whole response is the JSON string
            json_str = raw_text.strip()

        parsed_response = json.loads(json_str)

        return raw_text, parsed_response
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Hugging Face LLM: {e}")
        raise RuntimeError(
            f"Failed to connect to Hugging Face LLM at {model_endpoint}: {e}"
        )
    except json.JSONDecodeError as e:
        logging.error(
            f"Error decoding JSON from Hugging Face LLM response: {e}. Raw text: {raw_text}"
        )
        raise RuntimeError(f"Hugging Face LLM responded with invalid JSON: {e}")
    except Exception as e:
        logging.error(f"Error calling Hugging Face LLM: {e}")
        raise RuntimeError(f"Failed to get response from Hugging Face LLM: {e}")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    try:
        response = model.generate_content(prompt)
        raw_response_text = response.text  # Keep the original raw text

        # Find the JSON block using a regular expression
        json_match = re.search(
            r"```(json)?\s*({.*})\s*```", raw_response_text, re.DOTALL
        )

        if json_match:
            # Extract the JSON string from the regex match
            json_str = json_match.group(2)
        else:
            # If no markdown fence is found, assume the whole response is the JSON string
            json_str = raw_response_text.strip()

        # Parse the extracted JSON string
        parsed_response = json.loads(json_str)

        # Return the ORIGINAL raw text and the parsed dictionary
        return raw_response_text, parsed_response
    except Exception as e:
        logging.error(f"Error calling Gemini LLM: {e}")
        raise RuntimeError(f"Failed to get response from Gemini LLM: {e}")


def _call_openai_llm(
    api_key: str, model_name: str, prompt: str, base_url: Optional[str] = None
) -> tuple[str, dict]:
    """
    Calls the OpenAI LLM with the given API key, model name, and prompt.
    Expects the LLM to return a JSON string with a 'suggestions' key.
    """
    # If base_url is provided (e.g., for local Ollama), API key might not be strictly required.
    # However, if it's a standard OpenAI endpoint, api_key is essential.
    if not api_key and not base_url:
        raise ValueError(
            "OpenAI API Key is not configured for a standard OpenAI endpoint."
        )
    if not model_name:
        raise ValueError("OpenAI Model Name is not configured.")

    client = openai.OpenAI(
        api_key=api_key, base_url=base_url
    )  # Use base_url if provided

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw_response_text = response.choices[0].message.content

        # Find the JSON block using a regular expression
        json_match = re.search(
            r"```(json)?\s*({.*})\s*```", raw_response_text, re.DOTALL
        )

        if json_match:
            # Extract the JSON string from the regex match
            json_str = json_match.group(2)
        else:
            # If no markdown fence is found, assume the whole response is the JSON string
            json_str = raw_response_text.strip()

        parsed_response = json.loads(json_str)
        return raw_response_text, parsed_response
    except openai.APIConnectionError as e:
        logging.error(f"Error connecting to OpenAI-compatible LLM at {base_url}: {e}")
        raise RuntimeError(
            f"LLMサーバーへの接続に失敗しました。URL（{base_url}）が正しいか、サーバーが起動しているか確認してください。"
        )
    except Exception as e:
        logging.error(f"Error calling OpenAI LLM: {e}")
        raise RuntimeError(f"Failed to get response from OpenAI LLM: {e}")


def call_llm(
    api_key: Optional[str],
    model_name: Optional[str],
    prompt: str,
    llm_provider: str,
    base_url: Optional[str] = None,
    huggingface_api_key: Optional[str] = None,
    huggingface_model_base_endpoint: Optional[str] = None,
    huggingface_model_id: Optional[str] = None,
) -> tuple[str, dict]:
    """
    Dispatches to the appropriate LLM client based on the llm_provider.
    """
    logging.info(
        f"[LLM] PROVIDER: {llm_provider}: Model: {model_name if model_name else huggingface_model_id}"
    )

    if llm_provider == "gemini":
        if not model_name:
            raise ValueError("Gemini Model Name is not configured.")
        return _call_gemini_llm(api_key, model_name, prompt)
    elif llm_provider == "chatgpt":
        # ChatGPT specific logic for model_name and base_url can be added here if needed
        # For now, it will use _call_openai_llm, which is compatible with OpenAI's API
        return _call_openai_llm(api_key, model_name, prompt)
    elif llm_provider == "other":
        # 'other' assumes an OpenAI-compatible API, thus uses _call_openai_llm
        if not base_url:
            raise ValueError("Base URL is required for 'other' LLM provider.")
        return _call_openai_llm(api_key, model_name, prompt, base_url)
    elif llm_provider == "huggingface":
        if not huggingface_model_base_endpoint:
            raise ValueError("Hugging Face Base Endpoint is not configured.")
        if not huggingface_model_id:
            raise ValueError("Hugging Face Model ID is not configured.")
        return _call_huggingface_llm(
            huggingface_api_key,
            huggingface_model_base_endpoint,
            huggingface_model_id,
            prompt,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}.")
