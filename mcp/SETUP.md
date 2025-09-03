# MCP Setup Guide

This guide explains how to set up and run each version of the MCP Chainlit application.

## Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables in `local.env`

## Claude Version (`app.py`)

### Setup
1. Get your Anthropic API key from [console.anthropic.com](https://console.anthropic.com)
2. Update `local.env`:
   ```bash
   export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
   ```

### Run
```bash
chainlit run app.py
```

## ChatGPT Version (`app_chatgpt.py`)

### Setup
1. Get your OpenAI API key from [platform.openai.com](https://platform.openai.com/api-keys)
2. Update `local.env`:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

### Run
```bash
chainlit run app_chatgpt.py
```

## Llama Version (`app_llama.py`)

### Setup
1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Start Ollama service:
   ```bash
   ollama serve
   ```
3. Check available models and pull if needed:
   ```bash
   # Check what models you have
   ollama list
   
   # Pull popular models if not available
   ollama pull llama3        # Standard Llama 3 model
   ollama pull llama3.1      # Llama 3.1 (if available)
   ollama pull llama3.2      # Latest Llama 3.2 (if available)
   ollama pull llama4        # Llama 4 (if available)
   ```
4. Update `local.env` to match your available model:
   ```bash
   export OLLAMA_BASE_URL="http://localhost:11434"
   export OLLAMA_MODEL="llama3"  # Change to match your available model
   ```

### Run
```bash
chainlit run app_llama.py
```

## Model Recommendations

### Llama Models
- **llama3** - Standard Llama 3 model, good balance of speed and quality
- **llama3.1** - Improved Llama 3.1 model (if available)
- **llama3.2** - Latest Llama 3.2 model (if available)
- **llama4** - Latest Llama 4 model (if available)

**Note**: Model availability depends on what you have installed. Use `ollama list` to see your available models.

### Tool Calling Note
The Llama version uses a simplified tool calling approach since open-source models may not have native function calling support like commercial APIs. Tool calling reliability may vary depending on the model size and prompt engineering.

## Troubleshooting

### Ollama Issues
- **Connection refused**: Make sure `ollama serve` is running
- **Model not found**: 
  - First check available models: `ollama list`
  - Update `OLLAMA_MODEL` in `local.env` to match an available model
  - Or pull the missing model: `ollama pull <model-name>`
- **Out of memory**: Try a smaller/simpler model variant

### API Key Issues
- Make sure your API keys are valid and have sufficient credits
- Check that environment variables are properly loaded

### MCP Connection Issues
- Ensure MCP servers are properly configured and running
- Check Chainlit MCP documentation for connection setup
