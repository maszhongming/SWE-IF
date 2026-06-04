# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import requests
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# Provider-specific imports
from google import genai
from google.genai.types import (
    HttpOptions,
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    SafetySetting
)
from anthropic import AnthropicVertex
from mistralai_gcp import MistralGoogleCloud

# Import model configurations
from .model_configs import get_model_config


def _convert_google_genai_to_openai_format(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert from Google GenAI format to OpenAI format (used by Claude, Mistral, and OpenRouter).
    
    Google GenAI format:
        {'role': 'user', 'parts': [{'text': 'Hello'}]}
        
    OpenAI format:
        {'role': 'user', 'content': 'Hello'}
    """
    openai_messages = []
    for msg in messages:
        # Convert role names
        role = 'assistant' if msg.get('role') == 'model' else msg.get('role', 'user')
        
        # Extract text from parts
        text = " ".join(part['text'] for part in msg.get('parts', []) if 'text' in part)
        
        if text:
            openai_messages.append({'role': role, 'content': text})
    
    return openai_messages


def _sanitize_model_name_for_filename(model_name: str) -> str:
    """
    Sanitize model name for use in filenames by replacing slashes with double underscores.
    
    Examples:
        'x-ai/grok-4' -> 'x-ai__grok-4'
        'anthropic/claude-3-sonnet' -> 'anthropic__claude-3-sonnet'
        'gpt-4' -> 'gpt-4' (no change)
    """
    return model_name.replace('/', '__')


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, Any]]) -> str:
        """Generate response from the model."""
        pass


class GeminiProvider(BaseProvider):
    """Google Gemini provider implementation."""
    
    def __init__(self, model: str, config: Dict[str, Any], system_instruction: Optional[str]):
        self.model = f"google/{model}"
        self.config_dict = config
        self.client = genai.Client(http_options=HttpOptions(api_version="v1"))
        
        # Use temperature from config or override
        temperature = config.get('temperature', 0.0)
        max_tokens = config.get('max_tokens', 32768)
        
        # Disable all safety settings
        safety_settings = [
            SafetySetting(category=cat, threshold=HarmBlockThreshold.BLOCK_NONE)
            for cat in HarmCategory if cat != HarmCategory.HARM_CATEGORY_UNSPECIFIED
        ]
        
        self.config = GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            safety_settings=safety_settings,
            system_instruction=system_instruction,
        )
    
    def generate(self, messages: List[Dict[str, Any]]) -> str:
        """Generate response using Gemini API."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=messages,
                config=self.config,
            )
            
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content.parts:
                    return "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
                else:
                    print(f"Generation finished but produced no content. Finish Reason: {candidate.finish_reason.name}")
                    return ""
            else:
                print(f"No candidates returned. Prompt feedback: {response.prompt_feedback}")
                return ""

        except Exception as e:
            print(f"An exception occurred during the API call: {e}")
            return ""


class ClaudeProvider(BaseProvider):
    """Anthropic Claude provider implementation via Vertex AI."""
    
    def __init__(self, model: str, config: Dict[str, Any], system_instruction: Optional[str]):
        self.config_dict = config
        self.system_instruction = system_instruction
        
        # Handle thinking mode
        if 'thinking_budget_tokens' in config:
            self.base_model = model.replace('-thinking', '')
            self.thinking_enabled = True
            self.thinking_budget_tokens = config['thinking_budget_tokens']
            self.max_tokens = config.get('max_tokens', 32768)
            self.temperature = config.get('temperature', 1.0)
        else:
            self.base_model = model
            self.thinking_enabled = False
            self.thinking_budget_tokens = 0
            self.max_tokens = config.get('max_tokens', 16384)
            self.temperature = config.get('temperature', 0.0)
        
        # Get project ID from environment
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set for Claude models")
            
        self.client = AnthropicVertex(region="us-east5", project_id=project_id)
    
    def generate(self, messages: List[Dict[str, Any]]) -> str:
        """Generate response using Claude API."""
        # Convert from Google GenAI format to OpenAI format
        openai_messages = _convert_google_genai_to_openai_format(messages)
        
        # Build API parameters
        params = {
            "model": self.base_model,
            "messages": openai_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        if self.system_instruction:
            params["system"] = self.system_instruction
            
        if self.thinking_enabled:
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens
            }
        
        # Stream for max_tokens > 16k
        with self.client.messages.stream(**params) as stream:
            response = ""
            for text in stream.text_stream:
                response += text
        return response


class MistralProvider(BaseProvider):
    """Mistral provider implementation via Vertex AI."""
    
    def __init__(self, model: str, config: Dict[str, Any], system_instruction: Optional[str]):
        self.model = model
        self.config_dict = config
        self.system_instruction = system_instruction
        self.max_tokens = config.get('max_tokens', 32768)
        self.temperature = config.get('temperature', 0.0)
        
        # Get project ID and region from environment
        self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.region = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
        # self.region = "europe-west4"
        
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set for Mistral models")
        
        # Initialize Mistral client
        self.client = MistralGoogleCloud(
            region=self.region,
            project_id=self.project_id
        )
    
    def generate(self, messages: List[Dict[str, Any]]) -> str:
        """Generate response using Mistral API via mistralai_gcp."""
        # Convert from Google GenAI format to OpenAI format
        openai_messages = _convert_google_genai_to_openai_format(messages)
        
        # Add system message if provided
        if self.system_instruction:
            openai_messages.insert(0, {"role": "system", "content": self.system_instruction})

        resp = self.client.chat.complete(
            model=self.model,
            messages=openai_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        if resp.choices and len(resp.choices) > 0:
            return resp.choices[0].message.content
        else:
            return ""


class OpenRouterProvider(BaseProvider):
    """OpenRouter provider implementation."""
    
    def __init__(self, model: str, config: Dict[str, Any], system_instruction: Optional[str]):
        self.model = model  # Keep original model name with slashes for API calls
        self.config_dict = config
        self.system_instruction = system_instruction
        self.max_tokens = config.get('max_tokens', 32768)
        self.temperature = config.get('temperature', 0.0)
        
        # Get API key from environment
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable must be set for OpenRouter models")
        
        # OpenRouter API endpoint
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Setup headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate(self, messages: List[Dict[str, Any]]) -> str:
        """Generate response using OpenRouter API."""
        # Convert from Google GenAI format to OpenAI format
        openai_messages = _convert_google_genai_to_openai_format(messages)
        
        # Add system message if provided
        if self.system_instruction:
            openai_messages.insert(0, {"role": "system", "content": self.system_instruction})
        
        # Build API payload
        payload = {
            "model": self.model,  # Use original model name with slashes
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": openai_messages,
            # Quantization settings is only applied for open-source models
            # "provider": {
            #     "quantizations": ["fp16"]
            # },
        }
        
        # Add reasoning parameter if specified in config
        if self.config_dict.get('reasoning'):
            payload["reasoning"] = self.config_dict['reasoning']
        
        try:
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                data=json.dumps(payload),
                timeout=600  # 10 minute timeout
            )
            response.raise_for_status()
            response_data = response.json()
            
            if response_data.get('choices'):
                choice = response_data['choices'][0]
                
                # Check for error object in the choice.
                if choice.get('error'):
                    print(f"Error from OpenRouter choice: {choice['error']}")
                    return "" # Return empty on error

                # Check for abnormal finish reason.
                finish_reason = choice.get('finish_reason')
                if finish_reason != 'stop':
                    print(f"Warning: Abnormal finish reason: '{finish_reason}'")

                return choice['message']['content']
            else:
                raise ValueError("No choices in response")

        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}")
            return "" # Return empty on error
        except (KeyError, json.JSONDecodeError) as e:
            print(f"Error parsing response: {e}")
            return "" # Return empty on error


class LLMInterface:
    """
    Unified LLM interface supporting multiple providers.
    
    Uses Google GenAI message format for all providers:
        [{'role': 'user', 'parts': [{'text': 'Hello'}]}]
    
    Model naming conventions:
    - Gemini: "gemini-2.5-flash", "gemini-3.1-pro-preview"
    - Claude: "claude-sonnet-4@20250514", "claude-sonnet-4@20250514-thinking"
    - Mistral: "mistral-small-2503"
    - OpenRouter: "x-ai/grok-4", "deepseek/deepseek-r1", etc.
    """
    
    # Provider registry
    PROVIDERS = {
        'gemini': GeminiProvider,
        'claude': ClaudeProvider,
        'mistral': MistralProvider,
        'openrouter': OpenRouterProvider,
    }
    
    def __init__(
        self,
        model: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize the LLM interface.
        
        Args:
            model: Model identifier (can include -thinking suffix for Claude)
            temperature: Generation temperature (None = use model default)
            max_tokens: Maximum tokens to generate (None = use model default)
            system_instruction: Optional system prompt that applies to all requests
        """
        self.model = model
        self.sanitized_model_name = _sanitize_model_name_for_filename(model)
        
        # Get model configuration
        self.model_config = get_model_config(model)
        
        # Override config with explicit parameters
        if temperature is not None:
            self.model_config['temperature'] = temperature
        if max_tokens is not None:
            self.model_config['max_tokens'] = max_tokens
        
        # Determine provider from model config
        provider_name = self.model_config['provider']
        provider_class = self.PROVIDERS.get(provider_name)
        
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        self.provider = provider_class(model, self.model_config, system_instruction)
        
        # Print configuration info
        print(f"Initialized {model} with max_tokens={self.model_config['max_tokens']}, temperature={self.model_config['temperature']}")
    
    def generate_response(self, contents: List[Dict[str, Any]]) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            contents: List of message dictionaries in Google GenAI format:
                     [{'role': 'user', 'parts': [{'text': 'Hello'}]}]
                     
        Returns:
            The model's response as a string.
        """
        if not contents:
            return ""
            
        try:
            return self.provider.generate(contents)
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
    
    def get_sanitized_model_name(self) -> str:
        """
        Get model name sanitized for use in filenames.
        
        Returns:
            Model name with slashes replaced by double underscores.
        """
        return self.sanitized_model_name


def create_llm_interface(
    model: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    system_instruction: Optional[str] = None,
) -> LLMInterface:
    """
    Factory function to create an LLM interface.
    
    Args:
        model: Model identifier (append "-thinking" for Claude thinking mode)
        temperature: Generation temperature (None = use model default)
        max_tokens: Maximum tokens to generate (None = use model default)
        system_instruction: Optional system prompt for all requests
        
    Examples:
        # Use model defaults
        llm = create_llm_interface("gemini-2.5-flash")
        
        # Override specific settings
        llm = create_llm_interface(
            "claude-sonnet-4@20250514",
            temperature=0.1,
            max_tokens=8192,
            system_instruction="You are a helpful coding assistant."
        )
        
        # Claude with thinking (automatically uses correct settings)
        llm = create_llm_interface("claude-sonnet-4@20250514-thinking")
        
        # Mistral model
        llm = create_llm_interface("mistral-small-2503")
        
        # OpenRouter models
        llm = create_llm_interface("x-ai/grok-4")
        llm = create_llm_interface("anthropic/claude-3-sonnet")
    """
    return LLMInterface(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_instruction=system_instruction,
    )


# Example Usage
if __name__ == "__main__":
    # Example 1: Use model defaults
    print("--- Using Model Defaults ---")
    gemini_llm = create_llm_interface("gemini-2.5-flash")
    
    # Example 2: Override specific settings
    print("--- Override Settings ---")
    claude_llm = create_llm_interface(
        "claude-sonnet-4@20250514",
        temperature=0.1,
        max_tokens=8192
    )
    
    # Example 3: Claude thinking mode (automatically configured)
    print("--- Claude Thinking Mode ---")
    claude_thinking = create_llm_interface("claude-sonnet-4@20250514-thinking")
    
    # Example 4: Mistral model
    print("--- Mistral Model ---")
    mistral_llm = create_llm_interface("mistral-small-2503")
    
    # Example 5: OpenRouter models
    print("--- OpenRouter Models ---")
    grok_llm = create_llm_interface("x-ai/grok-4")
    or_claude_llm = create_llm_interface("anthropic/claude-3-sonnet")
    
    # Example 6: Test sanitized model names
    print("--- Sanitized Model Names ---")
    print(f"Original: 'x-ai/grok-4' -> Sanitized: '{_sanitize_model_name_for_filename('x-ai/grok-4')}'")
    print(f"Original: 'anthropic/claude-3-sonnet' -> Sanitized: '{_sanitize_model_name_for_filename('anthropic/claude-3-sonnet')}'")