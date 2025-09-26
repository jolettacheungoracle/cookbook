import json
import os
from dotenv import load_dotenv

from mcp import ClientSession
import chainlit as cl

from config_loader import get_model_config, get_current_model
from model_adapter import get_model_adapter

# Load environment variables from .env file
load_dotenv()

def get_system_prompt(supports_tools: bool = True):
    """Generate system prompt based on model capabilities"""
    base_prompt = """You are an expert GPU troubleshooting assistant specializing in NVIDIA GPU systems on Oracle Cloud Infrastructure (OCI). Your primary role is to help diagnose and resolve issues with BM.GPU.H100.8, BM.GPU.H200.8, BM.GPU.B200.8, BM.GPU.GB200.4 and BM.GPU.MI300X.8 instances.

## COMMUNICATION STYLE
Be concise and direct in your responses. Provide clear, actionable information without unnecessary verbosity."""
    
    if supports_tools:
        tool_guidance = """

## USER APPROVAL REQUIRED
IMPORTANT: Before running any MCP tools, you MUST request explicit approval from the user. Always ask the user for permission before executing any tool calls. For example:
- "I would like to run [tool_name] to [brief description of what it does]. May I proceed?"
- "To help diagnose this issue, I need to use [tool_name] which will [brief explanation]. Is that okay?"
Never execute MCP tools without first getting user consent.

## TOOL-FIRST APPROACH
Instead of just providing diagnostic commands as text, you should use the available MCP tools to actually perform diagnostics, run commands, and gather system information. Always prefer using tools over giving text instructions.

## TOOL USAGE FORMAT
When you need to use a tool, use this exact format:
TOOL_CALL: tool_name(parameter1="value1", parameter2="value2")

For example:
TOOL_CALL: check_monitoring_ring_health(monitoring_ring_id="f0258b6685684c113bad94d91b8fa02a", check_health="true")

## IMPORTANT: DO NOT PROVIDE COMMAND LINE SUGGESTIONS
Never suggest or recommend specific command line commands for the user to run manually. Always use the available MCP tools to perform any necessary operations instead of asking the user to execute commands themselves."""
    else:
        tool_guidance = """

## DIAGNOSTIC APPROACH
Since direct tool execution is not available, provide detailed diagnostic guidance including:
- Specific command line instructions for the user to run
- Clear explanations of what each command does and what to look for in the output
- Step-by-step troubleshooting procedures
- How to interpret results and determine next steps

## COMMAND LINE RECOMMENDATIONS
Provide specific commands for the user to execute, explaining:
- What each command does
- What output to expect
- How to interpret the results
- When to proceed to the next step"""
    
    return base_prompt + tool_guidance

# Additional knowledge base section that applies to all models
KNOWLEDGE_BASE = """

When troubleshooting:
1. Check system status, run diagnostics, and gather information
2. Use diagnostics to perform system checks, GPU diagnostics, and network verification  
3. Reference the knowledge base below to interpret results and determine next steps
4. Resolution steps need to be run in the order of the steps provided in the knowledge base

If you need OCI tools access, ask: "I need a compartment_id to access OCI resources. Please provide your OCI compartment_id."

## GPU TROUBLESHOOTING KNOWLEDGE BASE

### COMMON SYSTEM ISSUES

**Eth0 Not Named Properly**
- Symptoms: MPI jobs fail to start because eth0 is missing
- Resolution: Reboot the node if not present

**WPA Authentication Issues**
- Symptoms: Failed wpa supplicant errors in syslog, NCCL test fails
- Resolution: If PAE state != AUTHENTICATED, restart OCA plugin

**NVIDIA Fabric Manager Failed**
- Symptoms: One node NCCL test fails
- Resolution: Restart nvidia-fabricmanager.service or reboot

**HCA Fatal Error**
- Symptoms: Job fails to start or crashes midway
- Resolution: Clear dmesg and reboot. If persists, return to OCI

**Hardware Fell Off the Bus**
- Symptoms: GPU, RDMA interface missing
- Resolution: Reboot. If components missing within a day, return to OCI

**PCIe Width Missing Lanes**
- Symptoms: Performance lower than expected
- Resolution: If not seeing expected lanes, reboot. If persists, return to OCI

### COMMON GPU ISSUES

**ECC Errors**
- Symptoms: Job fails or crashes with memory error
- Resolution: If Volatile DRAM Uncorrectable errors, reboot. If SRAM Threshold Exceeded = Yes, return to OCI

**Row Remap Error**
- Symptoms: Job fails with row remap error
- Resolution: If last column != zero, return to OCI. If zero, reboot or reset GPUs

**Thermal Throttling**
- Symptoms: Performance drops, high GPU memory temperatures
- Resolution: If thermal throttling active, reboot with power action

**GPU Count Check Failure**
- Symptoms: nvidia-smi doesn't report 8 GPUs
- Resolution: 
1. Reboot. 
2. Confirm reboot via OCI
3. Rerun active health check for the instance
4. Check the logs of the active health check 
5. If the logs are successful, Resolution is complete.
6. If the logs show <8 GPUs, instruct to return the GPU to OCI. For an issue ticket provide the instance id and summary of the issue. Resolution is complete.

**NVLink Errors**
- Symptoms: Fabric manager fails, NCCL jobs fail
- Resolution: If any inactive links, reboot. If persists within a day, return to OCI

### COMMON NETWORK ISSUES

**RDMA Link Down**
- Symptoms: Job crashes or fails to start
- Resolution: If any != "Active", reboot. If still down, return to OCI

**RDMA Link Flaps**
- Symptoms: Job crashes, ib_create_qp/ib_modify_qp failures
- Resolution: Wait 30 minutes before checking. If >1 flap/hour or 3 total, investigate

**Source Based Routing Issues**
- Symptoms: Cannot communicate with other GPU hosts
- Resolution: If unexpected results, restart OCA plugin

### ESCALATION CRITERIA
- **Return to OCI**: Hardware failures, persistent issues after reboot, missing components
- **Contact OCI Support**: Authentication issues, GPUFM tickets needed, IB PKeys problems

Always use available tools to perform these checks and gather real data before making recommendations."""

def flatten(xss):
    return [x for xs in xss for x in xs]

def format_assistant_message_for_api(response, model_type="openai_compatible"):
    """Format assistant response for API compatibility (OpenAI or Anthropic)"""
    if model_type == "anthropic":
        # For Anthropic, return the response content as-is (it's already in the right format)
        return {"role": "assistant", "content": response.content}
    
    # For OpenAI-compatible APIs, convert to OpenAI format
    text_content = None
    tool_calls = []
    
    if hasattr(response, 'content') and isinstance(response.content, list):
        for block in response.content:
            if hasattr(block, 'type'):
                if block.type == "text" and hasattr(block, 'text'):
                    text_content = block.text
                elif block.type == "tool_use":
                    # Convert to OpenAI tool call format
                    tool_calls.append({
                        "id": getattr(block, 'id', f"call_{hash(block.name) % 10000}"),
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(getattr(block, 'input', {}))
                        }
                    })
    elif hasattr(response, 'content') and isinstance(response.content, str):
        text_content = response.content
    
    # Build the assistant message
    message = {"role": "assistant"}
    
    if text_content:
        message["content"] = text_content
    else:
        message["content"] = None
    
    if tool_calls:
        message["tool_calls"] = tool_calls
    
    return message

def format_tool_result_message(tool_use, tool_result, model_type="openai_compatible"):
    """Format tool result message for API compatibility (OpenAI or Anthropic)"""
    if model_type == "anthropic":
        # Anthropic format
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": str(tool_result),
                }
            ],
        }
    else:
        # OpenAI format
        return {
            "role": "tool",
            "tool_call_id": tool_use.id,
            "content": str(tool_result),
        }

def is_mcp_tool(tool_name):
    """Check if a tool is an MCP tool by looking it up in the mcp_tools registry"""
    mcp_tools = cl.user_session.get("mcp_tools", {})
    for connection_name, tools in mcp_tools.items():
        if any(tool.get("name") == tool_name for tool in tools):
            return True
    return False

@cl.on_mcp_connect
async def on_mcp(connection, session: ClientSession):
    result = await session.list_tools()
    tools = [{
        "name": t.name,
        "description": t.description,
        "input_schema": t.inputSchema,
        } for t in result.tools]
    
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_tools[connection.name] = tools
    cl.user_session.set("mcp_tools", mcp_tools)

@cl.step(type="tool") 
async def call_tool(tool_use):
    tool_name = tool_use.name
    tool_input = tool_use.input
    
    current_step = cl.context.current_step
    current_step.name = tool_name
    
    # Identify which mcp is used
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_name = None

    for connection_name, tools in mcp_tools.items():
        if any(tool.get("name") == tool_name for tool in tools):
            mcp_name = connection_name
            break
    
    if not mcp_name:
        current_step.output = json.dumps({"error": f"Tool {tool_name} not found in any MCP connection"})
        return current_step.output
    
    mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)
    
    if not mcp_session:
        current_step.output = json.dumps({"error": f"MCP {mcp_name} not found in any MCP connection"})
        return current_step.output
    
    try:
        current_step.output = await mcp_session.call_tool(tool_name, tool_input)
    except Exception as e:
        current_step.output = json.dumps({"error": str(e)})
    
    return current_step.output

async def call_model(chat_messages):
    """Call the currently configured model"""
    msg = cl.Message(content="")
    
    # Get current model configuration
    current_model = get_current_model()
    config = get_model_config(current_model)
    
    if not config:
        await cl.Message(content=f"❌ Model configuration not found for: {current_model}").send()
        return None
    
    try:
        # Get model adapter
        adapter = get_model_adapter(current_model)
        
        # Get MCP tools if available
        mcp_tools = cl.user_session.get("mcp_tools", {})
        tools = flatten([tools for _, tools in mcp_tools.items()]) if config.supports_tools else None
        
        # Generate dynamic system prompt based on model capabilities
        system_prompt = get_system_prompt(config.supports_tools) + KNOWLEDGE_BASE
        
        # Stream the response
        stream = await adapter.stream_chat(
            messages=chat_messages,
            tools=tools,
            system=system_prompt
        )
        
        response = await adapter.process_streaming_response(stream, msg)
        return response
        
    except Exception as e:
        error_msg = f"❌ Error with model {config.display_name}: {str(e)}"
        await cl.Message(content=error_msg).send()
        return None

@cl.on_chat_start
async def start_chat():
    cl.user_session.set("chat_messages", [])
    cl.user_session.set("pending_tool_approval", None)
    
    # Get current model info
    current_model = get_current_model()
    config = get_model_config(current_model)
    model_display = config.display_name if config else current_model
    
    # Send welcome message
    welcome_message = f"""👋 **Welcome to the GPU Troubleshooting Assistant!**

🤖 **Current Model:** {model_display}

I'm here to help you diagnose and resolve issues with your NVIDIA GPU systems.

**Are you experiencing any issues with your GPUs?** 

Please describe any problems you're facing, such as:
🔧 Performance issues or slowdowns
⚠️ Error messages or crashes
🌡️ Temperature/thermal concerns
🔗 Network connectivity problems
💾 Memory errors
🖥️ Missing GPUs or hardware components

I can run diagnostic tools to help identify and resolve the issue. Just let me know what's happening!"""
    
    await cl.Message(content=welcome_message).send()

@cl.on_message
async def on_message(msg: cl.Message):   
    chat_messages = cl.user_session.get("chat_messages")
    pending_approval = cl.user_session.get("pending_tool_approval")
    
    # Check if we're waiting for approval on a tool
    if pending_approval:
        user_response = msg.content.lower().strip()
        if user_response in ['yes', 'y', 'ok', 'okay', 'approve', 'proceed', 'go ahead']:
            # User approved - execute the tool
            tool_use = pending_approval["tool_use"]
            response = pending_approval["response"]
            
            await cl.Message(content="✅ Tool approved. Executing...").send()
            tool_result = await call_tool(tool_use)

            # Get current model configuration to determine format
            config = get_model_config()
            model_type = config.type if config else "openai_compatible"
            
            # Format messages for the current model type
            assistant_message = format_assistant_message_for_api(response, model_type)
            tool_result_message = format_tool_result_message(tool_use, tool_result, model_type)
            
            messages = [assistant_message, tool_result_message]

            chat_messages.extend(messages)
            cl.user_session.set("pending_tool_approval", None)
            
            # Continue processing
            response = await call_model(chat_messages)
            if response:
                await process_model_response(response, chat_messages)
            
        elif user_response in ['no', 'n', 'deny', 'decline', 'cancel', 'stop']:
            # User declined - inform model
            tool_use = pending_approval["tool_use"]
            response = pending_approval["response"]
            
            await cl.Message(content="❌ Tool execution declined by user.").send()
            
            # Get current model configuration to determine format
            config = get_model_config()
            model_type = config.type if config else "openai_compatible"
            
            # Format messages for the current model type
            assistant_message = format_assistant_message_for_api(response, model_type)
            
            # Create a mock tool result for the declined case
            class MockToolResult:
                def __str__(self):
                    return "Tool execution was declined by the user."
            
            tool_result_message = format_tool_result_message(tool_use, MockToolResult(), model_type)
            
            messages = [assistant_message, tool_result_message]

            chat_messages.extend(messages)
            cl.user_session.set("pending_tool_approval", None)
            
            # Continue processing
            response = await call_model(chat_messages)
            if response:
                await process_model_response(response, chat_messages)
            
        else:
            # Unclear response - ask for clarification
            await cl.Message(content="Please respond with 'yes' to approve the tool execution or 'no' to decline.").send()
        
        return
    
    # Handle uploaded files
    content = msg.content
    if msg.elements:
        file_contents = []
        for element in msg.elements:
            if hasattr(element, 'path') and element.path:
                try:
                    # Read file content
                    with open(element.path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    file_contents.append(f"File: {element.name}\n\nContent:\n{file_content}")
                except UnicodeDecodeError:
                    # Try reading as binary for non-text files
                    try:
                        with open(element.path, 'rb') as f:
                            file_data = f.read()
                        file_contents.append(f"File: {element.name}\n\nBinary file detected (size: {len(file_data)} bytes)")
                    except Exception as e:
                        file_contents.append(f"File: {element.name}\n\nError reading file: {str(e)}")
                except Exception as e:
                    file_contents.append(f"File: {element.name}\n\nError reading file: {str(e)}")
        
        if file_contents:
            content = f"{msg.content}\n\n--- Uploaded Files ---\n" + "\n\n".join(file_contents)
    
    chat_messages.append({"role": "user", "content": content})
    response = await call_model(chat_messages)
    
    if response:
        await process_model_response(response, chat_messages)

async def process_model_response(response, chat_messages):
    """Process model response, handling tool approvals"""
    # Check if this model supports tools and if tools were used
    config = get_model_config()
    
    # For models that support tools (like Anthropic)
    if config.supports_tools and hasattr(response, 'stop_reason') and response.stop_reason == "tool_use":
        tool_use = next(block for block in response.content if hasattr(block, 'type') and block.type == "tool_use")
        
        # Check if this is an MCP tool that needs approval
        if is_mcp_tool(tool_use.name):
            # Present for approval
            tool_description = f"**Tool:** `{tool_use.name}`\n**Purpose:** {tool_use.input if tool_use.input else 'Execute tool operation'}"
            approval_msg = f"🔧 **MCP Tool Approval Required**\n\n{tool_description}\n\nDo you approve running this tool? (yes/no)"
            
            await cl.Message(content=approval_msg).send()
            
            # Store pending approval
            cl.user_session.set("pending_tool_approval", {
                "tool_use": tool_use,
                "response": response
            })
            return
        else:
            # Non-MCP tool - execute directly
            tool_result = await call_tool(tool_use)

            # Get current model configuration to determine format
            config = get_model_config()
            model_type = config.type if config else "openai_compatible"
            
            # Format messages for the current model type
            assistant_message = format_assistant_message_for_api(response, model_type)
            tool_result_message = format_tool_result_message(tool_use, tool_result, model_type)
            
            messages = [assistant_message, tool_result_message]

            chat_messages.extend(messages)
            response = await call_model(chat_messages)
            if response:
                await process_model_response(response, chat_messages)
            return

    # Get final response text
    final_response = None
    if hasattr(response, 'content'):
        if isinstance(response.content, list):
            final_response = next(
                (block.text for block in response.content if hasattr(block, "text")),
                None,
            )
        elif isinstance(response.content, str):
            final_response = response.content
    
    if final_response:
        chat_messages.append({"role": "assistant", "content": final_response})
