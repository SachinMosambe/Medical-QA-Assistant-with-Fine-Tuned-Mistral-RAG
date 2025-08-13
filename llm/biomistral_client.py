
import os
from typing import Optional

import requests
from huggingface_hub import InferenceClient

class BioMistralClient:
    """
    A thin wrapper around HF Inference for BioMistral, supporting either:
    - A dedicated inference endpoint via HF_INFERENCE_URL (recommended for reliability)
    - Or the public Inference API with model id HF_MODEL_ID (default: BioMistral/BioMistral-7B)
    """
    def __init__(self, timeout: int = 60):
        self.api_token = os.getenv("HF_API_TOKEN")
        if not self.api_token:
            raise RuntimeError("HF_API_TOKEN is required. Set it in your environment or .env file.")
        self.endpoint = os.getenv("HF_INFERENCE_URL", "").strip()
        self.model_id = os.getenv("HF_MODEL_ID", "BioMistral/BioMistral-7B")
        self.timeout = timeout
        if not self.endpoint:
            self.client = InferenceClient(model=self.model_id, token=self.api_token, timeout=timeout)
        else:
            self.client = None  # we'll use raw HTTP to the endpoint
    
    def generate(self, prompt: str, temperature: float = 0.2, max_new_tokens: int = 512) -> str:
        if self.endpoint:
            headers = {"Authorization": f"Bearer {self.api_token}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "return_full_text": False
                }
            }
            resp = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"]
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"]
            if isinstance(data, list) and data and "output_text" in data[0]:
                return data[0]["output_text"]
            return str(data)
        else:
            return self.client.text_generation(
                prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                return_full_text=False
            )
