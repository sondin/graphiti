"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
import os
import typing
from typing import TYPE_CHECKING

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..prompts.models import Message
from .client import LLMClient
from .config import LLMConfig, ModelSize
from .errors import RateLimitError

if TYPE_CHECKING:
    pass
else:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(
            'openai is required for QwenClient. '
            'Install it with: pip install graphiti-core[openai]'
        ) from None

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'qwen-plus'
DEFAULT_MAX_TOKENS = 2048

# Qwen models with their context window sizes
QWEN_MODELS = {
    'qwen-turbo': 131072,  # 128K
    'qwen-plus': 131072,   # 128K
    'qwen-max': 32768,     # 32K
    'qwen-long': 32768,    # 32K
}


class QwenClient(LLMClient):
    """
    QwenClient is a client class for interacting with Qwen language models via DashScope API.

    This class extends the LLMClient and provides methods to initialize the client
    and generate responses from the Qwen language model using DashScope's OpenAI-compatible API.

    Attributes:
        client (AsyncOpenAI): The OpenAI-compatible client used to interact with DashScope API.
        model (str): The model name to use for generating responses.
        
    Methods:
        __init__(config: LLMConfig | None = None, cache: bool = False):
            Initializes the QwenClient with the provided configuration and cache setting.

        _generate_response(messages: list[Message], response_model: type[BaseModel] | None = None, 
                          max_tokens: int = DEFAULT_MAX_TOKENS, model_size: ModelSize = ModelSize.medium) -> dict[str, typing.Any]:
            Generates a response from the language model based on the provided messages.
    """

    def __init__(self, config: LLMConfig | None = None, cache: bool = False):
        """
        Initialize the QwenClient with the provided configuration and cache setting.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, temperature, and max tokens.
                                       If no API key is provided, it will try to read from the DASHSCOPE_API_KEY environment variable.
            cache (bool): Whether to use caching for responses. Defaults to False.
        """
        if config is None:
            config = LLMConfig(max_tokens=DEFAULT_MAX_TOKENS)
        elif config.max_tokens is None:
            config.max_tokens = DEFAULT_MAX_TOKENS
            
        super().__init__(config, cache)

        # Use DASHSCOPE_API_KEY environment variable if no API key is provided
        api_key = config.api_key or os.getenv('DASHSCOPE_API_KEY')
        if not api_key:
            raise ValueError(
                'QwenClient requires an API key. '
                'Provide it in the LLMConfig or set the DASHSCOPE_API_KEY environment variable.'
            )

        # Initialize the OpenAI-compatible client for DashScope
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )

    def _convert_messages_to_openai_format(
        self, messages: list[Message]
    ) -> list[ChatCompletionMessageParam]:
        """Convert internal Message format to OpenAI ChatCompletionMessageParam format."""
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': m.content})
        return openai_messages

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Generate a response from the Qwen language model.

        Args:
            messages (list[Message]): A list of messages to send to the language model.
            response_model (type[BaseModel] | None): An optional Pydantic model to parse the response into.
            max_tokens (int): The maximum number of tokens to generate in the response.
            model_size (ModelSize): The size of the model to use (small or medium).

        Returns:
            dict[str, typing.Any]: The response from the language model.

        Raises:
            RateLimitError: If the API rate limit is exceeded.
            Exception: If there is an error generating the response.
        """
        openai_messages = self._convert_messages_to_openai_format(messages)
        
        # For Qwen models, we need to add a special indicator for JSON responses
        if response_model is not None:
            # Add explicit instruction for JSON output for Qwen models
            if openai_messages and openai_messages[-1]['role'] == 'user':
                openai_messages[-1]['content'] += '\n\nRespond with a JSON object.'
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model or DEFAULT_MODEL,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                response_format={'type': 'json_object'} if response_model else None,
            )
            
            result = response.choices[0].message.content or '{}'
            
            # For Qwen models, we may need to handle cases where it adds extra text around JSON
            if response_model is not None:
                # Try to extract JSON if there's extra text
                result_stripped = result.strip()
                if result_stripped.startswith('```json'):
                    result_stripped = result_stripped[7:]
                if result_stripped.endswith('```'):
                    result_stripped = result_stripped[:-3]
                result_stripped = result_stripped.strip()
                
                try:
                    return json.loads(result_stripped)
                except json.JSONDecodeError:
                    # If JSON parsing fails, return the raw result
                    return {'content': result}
            else:
                return {'content': result}
                
        except Exception as e:
            # Handle specific Qwen/DashScope errors
            error_message = str(e).lower()
            if 'rate limit' in error_message or '429' in str(e):
                raise RateLimitError from e
            
            logger.error(f'Error in generating Qwen response: {e}')
            raise