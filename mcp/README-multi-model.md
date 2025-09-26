# Multi-Model GPU Troubleshooting Assistant

This is an enhanced version of the GPU troubleshooting assistant that supports multiple AI models, allowing you to easily switch between different language models like Anthropic Claude and Llama 4 using environment variables.

## 🚀 Quick Start

1. **Set up environment variables:**
   ```bash
   cp env.example local.env
   # Edit .env with your API keys and set CURRENT_MODEL
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the multi-model app:**
   ```bash
   chainlit run app-multi-model.py
   ```

## 🎛️ Model Configuration

### Setting the Current Model

Set your preferred model by editing the `.env` file:

```bash
# In .env file
CURRENT_MODEL=llama4      # Use Llama 4
# or
CURRENT_MODEL=anthropic   # Use Anthropic Claude
```

### Available Models

The system comes pre-configured with:

- **anthropic** - Anthropic Claude (supports function calling)
- **llama4** - Llama 4 Maverick via OpenAI-compatible API

## ⚙️ Adding New Models

### Step 1: Create Model Configuration

Create a new JSON file in the `config/` directory:

```json
{
  "name": "my_model",
  "display_name": "My Custom Model",
  "type": "openai_compatible",
  "model": "my-model-name",
  "api_key_env": "MY_MODEL_API_KEY",
  "base_url": "https://api.mymodel.com/v1",
  "max_tokens": 1024,
  "temperature": 0.7,
  "streaming": true,
  "supports_tools": false,
  "description": "My custom model description"
}
```

### Step 2: Set Environment Variables

Add to your `.env` file:
```bash
MY_MODEL_API_KEY=your_api_key_here
CURRENT_MODEL=my_model
```

### Step 3: Restart the Application

```bash
chainlit run app-multi-model.py
```

## 📁 File Structure

```
mcp/
├── app-multi-model.py      # Main multi-model application
├── app-anthropic.py        # Original Anthropic-only app
├── config/                 # Model configurations
│   ├── anthropic.json     # Anthropic Claude config
│   └── llama4.json        # Llama 4 config
├── config_loader.py        # Configuration management
├── model_adapter.py        # Model API adapters
├── setup_multi_model.py    # Setup script
├── env.example             # Environment template
└── requirements.txt        # Dependencies
```

## 🔧 Configuration Options

### Model Configuration Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `name` | Unique model identifier | ✅ |
| `display_name` | Human-readable name | ✅ |
| `type` | API type (`anthropic` or `openai_compatible`) | ✅ |
| `model` | Model name/ID for the API | ✅ |
| `api_key_env` | Environment variable name for API key | ✅ |
| `base_url` | Custom API endpoint (for OpenAI-compatible) | ❌ |
| `max_tokens` | Maximum response tokens | ❌ |
| `temperature` | Sampling temperature | ❌ |
| `streaming` | Enable streaming responses | ❌ |
| `supports_tools` | Whether model supports function calling | ❌ |
| `description` | Model description | ❌ |

### Supported API Types

#### `anthropic`
- Uses the Anthropic SDK
- Supports function calling/tools
- Requires `ANTHROPIC_API_KEY`

#### `openai_compatible`
- Uses OpenAI SDK with custom endpoints
- Compatible with various models (Llama, etc.)
- May or may not support function calling

## 🛠️ Development

### Adding New API Types

1. Create a new adapter class in `model_adapter.py`:
   ```python
   class MyAPIAdapter(BaseModelAdapter):
       def _initialize_client(self):
           # Initialize your API client
           pass
       
       async def stream_chat(self, messages, tools=None, system=None):
           # Implement streaming chat
           pass
       
       async def process_streaming_response(self, stream, msg):
           # Process the streaming response
           pass
   ```

2. Register it in the `ModelAdapterFactory`:
   ```python
   elif config.type == "my_api":
       return MyAPIAdapter(config)
   ```

### Testing Models

```bash
# Run setup to test configuration
python setup_multi_model.py
```

## 🚨 Troubleshooting

### Common Issues

1. **Model not found error:**
   - Check that the model config file exists in `config/`
   - Verify the `CURRENT_MODEL` value in `.env` matches a config filename

2. **Missing API key error:**
   - Check that the environment variable is set in `.env`
   - Verify the variable name matches `api_key_env` in the config

3. **Connection errors:**
   - Verify the `base_url` in the configuration
   - Check if the API endpoint is accessible

### Debug Mode

Set `DEBUG=True` in your `.env` file for detailed logging.

## 📋 Dependencies

Required packages:
```
chainlit
anthropic
openai
python-dotenv
```

Install with:
```bash
pip install chainlit anthropic openai python-dotenv
```

## 🔄 Migration from Original App

To migrate from the original `app-anthropic.py`:

1. Copy your `.env` file
2. Install additional dependencies: `pip install openai`
3. Add `CURRENT_MODEL=anthropic` to your `.env` file
4. Use `app-multi-model.py` instead of `app-anthropic.py`

The new app is fully backward compatible with the original functionality.

## 🔧 Environment Variables

Your `.env` file should contain:

```bash
# Model selection (anthropic or llama4)
CURRENT_MODEL=anthropic

# Anthropic API
ANTHROPIC_API_KEY=your_anthropic_key_here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# OpenAI-compatible APIs (for Llama4, etc.)
OPENAI_API_KEY=dummy-key

# Optional Chainlit settings
CHAINLIT_AUTH_SECRET=your_secret_here
DEBUG=False
```

Change `CURRENT_MODEL` to switch between models and restart the application.