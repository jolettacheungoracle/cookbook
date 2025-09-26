import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, AsyncGenerator, Optional

import anthropic
import openai
import chainlit as cl

from config_loader import ModelConfig


class BaseModelAdapter(ABC):
    """Abstract base class for model adapters"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = None
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the API client"""
        pass
    
    @abstractmethod
    async def stream_chat(self, messages: List[Dict], tools: List[Dict] = None, system: str = None) -> Any:
        """Stream chat completion with the model"""
        pass
    
    @abstractmethod
    async def process_streaming_response(self, stream, msg: cl.Message) -> Any:
        """Process the streaming response and update the message"""
        pass


class AnthropicAdapter(BaseModelAdapter):
    """Adapter for Anthropic Claude models"""
    
    def _initialize_client(self):
        api_key = self.config.get_api_key()
        if not api_key:
            raise ValueError(f"API key not found for {self.config.api_key_env}")
        
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
    
    async def stream_chat(self, messages: List[Dict], tools: List[Dict] = None, system: str = None) -> Any:
        """Stream chat completion with Anthropic Claude"""
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": messages,
        }
        
        if system:
            kwargs["system"] = system
        
        if tools and self.config.supports_tools:
            kwargs["tools"] = tools
        
        return self.client.messages.stream(**kwargs)
    
    async def process_streaming_response(self, stream, msg: cl.Message) -> Any:
        """Process Anthropic streaming response"""
        async with stream as stream_context:
            async for text in stream_context.text_stream:
                await msg.stream_token(text)
        
        await msg.send()
        return await stream_context.get_final_message()


class OpenAICompatibleAdapter(BaseModelAdapter):
    """Adapter for OpenAI-compatible APIs (like Llama models)"""
    
    def _initialize_client(self):
        api_key = self.config.get_api_key() or "dummy-key"  # Some endpoints don't require real keys
        
        kwargs = {
            "api_key": api_key,
        }
        
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        
        self.client = openai.AsyncOpenAI(**kwargs)
    
    def _format_tools_for_system_message(self, tools: List[Dict]) -> str:
        """Format MCP tools for inclusion in system message"""
        if not tools:
            return ""
        
        tool_descriptions = []
        for tool in tools:
            name = tool.get("name", "unknown")
            description = tool.get("description", "No description available")
            
            # Format parameters if available
            input_schema = tool.get("input_schema", {})
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])
            
            param_info = []
            for param_name, param_def in properties.items():
                param_type = param_def.get("type", "string")
                param_desc = param_def.get("description", "")
                is_required = param_name in required
                req_text = " (required)" if is_required else " (optional)"
                param_info.append(f"  - {param_name} ({param_type}){req_text}: {param_desc}")
            
            params_text = "\n".join(param_info) if param_info else "  No parameters"
            
            tool_descriptions.append(f"• {name}: {description}\n{params_text}")
        
        return "\n\n".join(tool_descriptions)
    
    def _convert_mcp_tools_to_openai_format(self, mcp_tools: List[Dict]) -> List[Dict]:
        """Convert MCP tools to OpenAI function calling format"""
        openai_tools = []
        
        for tool in mcp_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", "unknown"),
                    "description": tool.get("description", ""),
                }
            }
            
            # Convert input_schema to parameters
            input_schema = tool.get("input_schema", {})
            if input_schema:
                # Copy the schema but rename it to 'parameters' for OpenAI format
                openai_tool["function"]["parameters"] = input_schema.copy()
            else:
                # Provide empty parameters if none specified
                openai_tool["function"]["parameters"] = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            
            openai_tools.append(openai_tool)
        
        return openai_tools
    
    def _parse_tool_call_format(self, content: str) -> Optional[Dict]:
        """Parse TOOL_CALL format from model response"""
        import re
        
        # Look for TOOL_CALL: tool_name(param1="value1", param2="value2") format
        pattern = r'TOOL_CALL:\s*(\w+)\((.*?)\)'
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        
        if not match:
            return None
        
        tool_name = match.group(1)
        params_str = match.group(2).strip()
        
        # Parse parameters
        args = {}
        if params_str:
            # Simple parameter parsing - handle key="value" pairs (including hyphens in keys)
            param_pattern = r'([\w-]+)\s*=\s*"([^"]*)"'
            for param_match in re.finditer(param_pattern, params_str):
                key = param_match.group(1)
                value = param_match.group(2)
                args[key] = value
        
        return {
            "name": tool_name,
            "args": args
        }
    
    async def stream_chat(self, messages: List[Dict], tools: List[Dict] = None, system: str = None) -> Any:
        """Stream chat completion with OpenAI-compatible API"""
        # Convert system message format if needed
        if system:
            # Add system message to the beginning if it doesn't exist
            if not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": system}] + messages
        
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": True,
        }
        
        # Add tools if supported by the model
        if tools and self.config.supports_tools:
            # Try with tools first, fall back if server doesn't support it
            try:
                # Convert MCP tools to OpenAI format
                openai_tools = self._convert_mcp_tools_to_openai_format(tools)
                kwargs["tools"] = openai_tools
                return await self.client.chat.completions.create(**kwargs)
            except Exception as e:
                # If server doesn't support function calling, try without tools
                # but include tool information in system message
                if "tool choice" in str(e).lower() or "enable-auto-tool-choice" in str(e).lower():
                    # Remove tools parameter and add tool descriptions to system message
                    kwargs.pop("tools", None)
                    
                    # Add tool descriptions to system message
                    tool_descriptions = self._format_tools_for_system_message(tools)
                    if tool_descriptions and kwargs.get("messages"):
                        # Find or create system message
                        messages = kwargs["messages"]
                        system_msg_idx = None
                        for i, msg in enumerate(messages):
                            if msg.get("role") == "system":
                                system_msg_idx = i
                                break
                        
                        tool_prompt = f"\n\nAVAILABLE TOOLS:\n{tool_descriptions}\n\nTo use a tool, respond with the exact format: TOOL_CALL: tool_name(parameter1=\"value1\", parameter2=\"value2\")"
                        
                        if system_msg_idx is not None:
                            messages[system_msg_idx]["content"] += tool_prompt
                        else:
                            messages.insert(0, {"role": "system", "content": f"You are a helpful assistant.{tool_prompt}"})
                    
                    return await self.client.chat.completions.create(**kwargs)
                else:
                    raise e
        
        return await self.client.chat.completions.create(**kwargs)
    
    async def process_streaming_response(self, stream, msg: cl.Message) -> Any:
        """Process OpenAI-compatible streaming response"""
        full_content = ""
        tool_calls = []
        finish_reason = "stop"
        
        async for chunk in stream:
            if chunk.choices:
                choice = chunk.choices[0]
                
                # Handle text content
                if choice.delta.content:
                    content = choice.delta.content
                    full_content += content
                    await msg.stream_token(content)
                
                # Handle tool calls
                if choice.delta.tool_calls:
                    for tool_call in choice.delta.tool_calls:
                        # Initialize tool call if not exists
                        while len(tool_calls) <= tool_call.index:
                            tool_calls.append({
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })
                        
                        current_tool_call = tool_calls[tool_call.index]
                        
                        if tool_call.id:
                            current_tool_call["id"] = tool_call.id
                        if tool_call.function:
                            if tool_call.function.name:
                                current_tool_call["function"]["name"] = tool_call.function.name
                            if tool_call.function.arguments:
                                current_tool_call["function"]["arguments"] += tool_call.function.arguments
                
                # Check finish reason
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
        
        await msg.send()
        
        # Check if the response contains TOOL_CALL format (fallback for servers that don't support function calling)
        tool_call_match = self._parse_tool_call_format(full_content)
        if tool_call_match:
            tool_calls.append({
                "id": f"call_{hash(tool_call_match['name']) % 10000}",
                "type": "function", 
                "function": {
                    "name": tool_call_match['name'],
                    "arguments": json.dumps(tool_call_match['args'])
                }
            })
            finish_reason = "tool_calls"
        
        # Create a mock response object similar to Anthropic's format
        class MockResponse:
            def __init__(self, content: str, tool_calls: list, finish_reason: str):
                self._content_blocks = []
                self._raw_content = content  # Store raw content for serialization
                
                # Add text content if present
                if content.strip():
                    self._content_blocks.append(MockTextContent(content))
                
                # Add tool calls if present
                for tool_call in tool_calls:
                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                    except (json.JSONDecodeError, KeyError):
                        args = {}
                    
                    self._content_blocks.append(MockToolContent(
                        id=tool_call.get("id", ""),
                        name=tool_call["function"]["name"],
                        input=args
                    ))
                
                # Set stop reason based on OpenAI finish reason
                if finish_reason == "tool_calls":
                    self.stop_reason = "tool_use"
                else:
                    self.stop_reason = "end_turn"
            
            @property
            def content(self):
                """Return content blocks list (consistent with Anthropic format)"""
                return self._content_blocks
            
            def __str__(self):
                """Return string representation for serialization"""
                return self._raw_content
            
            def __repr__(self):
                return f"MockResponse(content_blocks={len(self._content_blocks)}, stop_reason='{self.stop_reason}')"
        
        class MockTextContent(dict):
            def __init__(self, text: str):
                super().__init__()
                self.text = text
                self.type = "text"
                # Make it behave like a dict for JSON serialization
                self["text"] = text
                self["type"] = "text"
            
            def __str__(self):
                return self.text
            
            def __repr__(self):
                return f"MockTextContent(text='{self.text}')"
        
        class MockToolContent(dict):
            def __init__(self, id: str, name: str, input: dict):
                super().__init__()
                self.id = id
                self.name = name
                self.input = input
                self.type = "tool_use"
                # Make it behave like a dict for JSON serialization
                self["id"] = id
                self["name"] = name  
                self["input"] = input
                self["type"] = "tool_use"
            
            def __str__(self):
                return f"Tool: {self.name}({self.input})"
            
            def __repr__(self):
                return f"MockToolContent(name='{self.name}', input={self.input})"
        
        return MockResponse(full_content, tool_calls, finish_reason)


class ModelAdapterFactory:
    """Factory class to create appropriate model adapters"""
    
    @staticmethod
    def create_adapter(config: ModelConfig) -> BaseModelAdapter:
        """Create the appropriate adapter based on model type"""
        if config.type == "anthropic":
            return AnthropicAdapter(config)
        elif config.type == "openai_compatible":
            return OpenAICompatibleAdapter(config)
        else:
            raise ValueError(f"Unsupported model type: {config.type}")


# Convenience function
def get_model_adapter(model_name: str = None) -> BaseModelAdapter:
    """Get a model adapter for the specified or current model"""
    from config_loader import get_model_config
    
    config = get_model_config(model_name)
    if not config:
        raise ValueError(f"Model configuration not found for: {model_name or 'current model'}")
    
    return ModelAdapterFactory.create_adapter(config)
