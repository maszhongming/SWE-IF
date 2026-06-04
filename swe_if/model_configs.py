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

""" Model configurations for LLM interface. Contains essential parameters for different models. """

MODEL_CONFIGS = {
    # Gemini models
    'gemini-3.1-pro-preview': {
        'provider': 'gemini',
        'max_tokens': 32768,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 512,
        }
    },
    'gemini-2.5-pro': {
        'provider': 'gemini',
        'max_tokens': 32768,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 512,
        }
    },
    'gemini-2.5-flash': {
        'provider': 'gemini',
        'max_tokens': 32768,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 512,
        }
    },
    'gemini-2.0-flash-001': {
        'provider': 'gemini',
        'max_tokens': 8192,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 128,
        }
    },
    'gemini-2.0-flash-lite': {
        'provider': 'gemini',
        'max_tokens': 8192,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 128,
        }
    },
    
    # Claude models (thinking mode)
    'claude-opus-4@20250514-thinking': {
        'provider': 'claude',
        'max_tokens': 32000,
        'temperature': 1.0,
        'thinking_budget_tokens': 24576,
        'concurrent': {
            'max_workers': 2,
        }
    },
    'claude-sonnet-4@20250514-thinking': {
        'provider': 'claude',
        'max_tokens': 32768,
        'temperature': 1.0,
        'thinking_budget_tokens': 24576,
        'concurrent': {
            'max_workers': 16,
        }
    },
    'claude-3-7-sonnet@20250219-thinking': {
        'provider': 'claude',
        'max_tokens': 32768,
        'temperature': 1.0,
        'thinking_budget_tokens': 24576,
        'concurrent': {
            'max_workers': 32,
        }
    },
    
    # Claude models (standard — 3.5/3 series do not support thinking)
    'claude-3-5-sonnet-v2@20241022': {
        'provider': 'claude',
        'max_tokens': 8192,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 16,
        }
    },
    'claude-3-5-haiku@20241022': {
        'provider': 'claude',
        'max_tokens': 8192,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 16,
        }
    },
    'claude-3-haiku@20240307': {
        'provider': 'claude',
        'max_tokens': 4096,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 4,
        }
    },
    
    # Models from OpenRouter
    # DeepSeek models
    'deepseek/deepseek-chat-v3-0324': {
        'provider': 'openrouter',
        'max_tokens': 32768,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 512,
        }
    },
    'deepseek/deepseek-r1-0528': {
        'provider': 'openrouter',
        'max_tokens': 32768,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 128,
        }
    },
    # OpenAI models
    'openai/gpt-5': {
        'provider': 'openrouter',
        'max_tokens': 32768,
        'temperature': 0.0,
        "reasoning": {
            "effort": "high"
        },
        'concurrent': {
            'max_workers': 512,
        }
    },
    'openai/o4-mini': {
        'provider': 'openrouter',
        'max_tokens': 32768,
        'temperature': 0.0,
        "reasoning": {
            "effort": "medium"
        },
        'concurrent': {
            'max_workers': 512,
        }
    },
    'openai/o3-mini-high': {
        'provider': 'openrouter',
        'max_tokens': 32768,
        'temperature': 0.0,
        "reasoning": {
            "effort": "high"
        },
        'concurrent': {
            'max_workers': 512,
        }
    },
    'openai/gpt-4.1': {
        'provider': 'openrouter',
        'max_tokens': 32768,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 512,
        }
    },
    'openai/gpt-4.1-mini': {
        'provider': 'openrouter',
        'max_tokens': 32768,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 512,
        }
    },
    'openai/gpt-4o-2024-08-06': {
        'provider': 'openrouter',
        'max_tokens': 32768,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 512,
        }
    },
    'openai/gpt-4o-mini': {
        'provider': 'openrouter',
        'max_tokens': 32768,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 512,
        }
    },
    # Grok models
    'x-ai/grok-4': {
        'provider': 'openrouter',
        'max_tokens': 32768,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 512,
        }
    },
    'x-ai/grok-3-mini-beta': {
        'provider': 'openrouter',
        'max_tokens': 32768,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 512,
        }
    },
    # Qwen models
    'qwen/qwen3-235b-a22b': {
        'provider': 'openrouter',
        'max_tokens': 32768,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 32,
        }
    },
    'qwen/qwen3-32b': {
        'provider': 'openrouter',
        'max_tokens': 32768,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 32,
        }
    },
    'qwen/qwen3-30b-a3b': {
        'provider': 'openrouter',
        'max_tokens': 32768,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 32,
        }
    },
    'qwen/qwen-2.5-72b-instruct': {
        'provider': 'openrouter',
        'max_tokens': 32768,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 32,
        }
    },
    'qwen/qwen-2.5-coder-32b-instruct': {
        'provider': 'openrouter',
        'max_tokens': 16384,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 32,
        }
    },
    # Mistral models
    'mistralai/mistral-medium-3': {
        'provider': 'openrouter',
        'max_tokens': 32768,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 64,
        }
    },
    # Kimi models
    'moonshotai/kimi-k2': {
        'provider': 'openrouter',
        'max_tokens': 32768,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 32,
        }
    },
    # Gemma models
    'google/gemma-3-27b-it': {
        'provider': 'openrouter',
        'max_tokens': 32768,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 128,
        }
    },
    'google/gemma-3-12b-it': {
        'provider': 'openrouter',
        'max_tokens': 32768,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 128,
        }
    },
    # Minimax models
    'minimax/minimax-m1': {
        'provider': 'openrouter',
        'max_tokens': 32768,
        'temperature': 0.0,
        'concurrent': {
            'max_workers': 64,
        }
    },
}


def get_model_config(model_name: str) -> dict:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Configuration dictionary for the model. A model not listed in
        MODEL_CONFIGS falls back to a generic OpenRouter config
        (provider=openrouter, max_tokens=32768, temperature=0.0), so any model
        served by OpenRouter works out of the box. Add an explicit entry to
        MODEL_CONFIGS above for precise control (provider, decoding, thinking).
    """
    if model_name not in MODEL_CONFIGS:
        print(
            f"[swe_if] '{model_name}' is not in MODEL_CONFIGS; falling back to "
            f"defaults (provider=openrouter, max_tokens=32768, temperature=0.0). "
            f"Add an entry to model_configs.py for precise control."
        )
        return {
            'provider': 'openrouter',
            'max_tokens': 32768,
            'temperature': 0.0,
        }

    return MODEL_CONFIGS[model_name].copy()


def get_concurrent_config(model_name: str) -> dict:
    """
    Get concurrent processing configuration for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Concurrent configuration dictionary with default values applied
        
    Raises:
        ValueError: If model is not supported
    """
    model_config = get_model_config(model_name)
    concurrent_config = model_config.get('concurrent', {})
    
    # Apply default values for common parameters
    defaults = {
        'rate_limit_delay': 0.05,
        'retry_delay': 2.0,
        'concurrent_max_retries': 1
    }
    
    # Merge model-specific config with defaults
    final_config = defaults.copy()
    final_config.update(concurrent_config)
    
    return final_config