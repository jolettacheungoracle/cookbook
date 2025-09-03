import json

from mcp import ClientSession
import anthropic

import chainlit as cl

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

anthropic_client = anthropic.AsyncAnthropic()
SYSTEM = """You are an expert GPU troubleshooting assistant specializing in NVIDIA GPU systems on Oracle Cloud Infrastructure (OCI). Your primary role is to help diagnose and resolve issues with BM.GPU.H100.8, BM.GPU.H200.8, BM.GPU.B200.8, and BM.GPU.GB200.4 instances.

## USER APPROVAL REQUIRED
IMPORTANT: Before running any MCP tools, you MUST request explicit approval from the user. Always ask the user for permission before executing any tool calls. For example:
- "I would like to run [tool_name] to [brief description of what it does]. May I proceed?"
- "To help diagnose this issue, I need to use [tool_name] which will [brief explanation]. Is that okay?"
Never execute MCP tools without first getting user consent.

## COMMUNICATION STYLE
Be concise and direct in your responses. Provide clear, actionable information without unnecessary verbosity.

## TOOL-FIRST APPROACH
Instead of just providing diagnostic commands as text, you should use the available MCP tools to actually perform diagnostics, run commands, and gather system information. Always prefer using tools over giving text instructions.

## IMPORTANT: DO NOT PROVIDE COMMAND LINE SUGGESTIONS
Never suggest or recommend specific command line commands for the user to run manually. Always use the available MCP tools to perform any necessary operations instead of asking the user to execute commands themselves.

When troubleshooting:
1. Use available MCP tools to check system status, run diagnostics, and gather information
2. Analyze the actual results from tools rather than just providing theoretical guidance
3. Use tools to perform system checks, GPU diagnostics, and network verification
4. Reference the knowledge base below to interpret results and determine next steps
5. Only provide manual commands as a last resort when no suitable tool is available

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

**Missing GPUs**
- Symptoms: nvidia-smi doesn't report 8 GPUs
- Resolution: 
1. Reboot. 
2. Confirm reboot via OCI
3. Rerun active health check for the instance
4. Check the logs of the active health check
5. If <8 GPUs within a day, return to OCI with the instance id and summary of the issue

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

async def call_claude(chat_messages):
    msg = cl.Message(content="")
    mcp_tools = cl.user_session.get("mcp_tools", {})
    # Flatten the tools from all MCP connections
    tools = flatten([tools for _, tools in mcp_tools.items()])
    print([tool.get("name") for tool in tools])
    async with anthropic_client.messages.stream(
        system=SYSTEM,
        max_tokens=1024,
        messages=chat_messages,
        tools=tools,
        model=os.getenv("ANTHROPIC_MODEL"),
    ) as stream:
        async for text in stream.text_stream:
            await msg.stream_token(text)
    
    await msg.send()
    response = await stream.get_final_message()

    return response

@cl.on_chat_start
async def start_chat():
    cl.user_session.set("chat_messages", [])
    cl.user_session.set("pending_tool_approval", None)


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

            messages = [
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": str(tool_result),
                        }
                    ],
                },
            ]

            chat_messages.extend(messages)
            cl.user_session.set("pending_tool_approval", None)
            
            # Continue processing with Claude
            response = await call_claude(chat_messages)
            await process_claude_response(response, chat_messages)
            
        elif user_response in ['no', 'n', 'deny', 'decline', 'cancel', 'stop']:
            # User declined - inform Claude
            tool_use = pending_approval["tool_use"]
            response = pending_approval["response"]
            
            await cl.Message(content="❌ Tool execution declined by user.").send()
            
            messages = [
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": "Tool execution was declined by the user.",
                        }
                    ],
                },
            ]

            chat_messages.extend(messages)
            cl.user_session.set("pending_tool_approval", None)
            
            # Continue processing with Claude
            response = await call_claude(chat_messages)
            await process_claude_response(response, chat_messages)
            
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
    response = await call_claude(chat_messages)
    
    await process_claude_response(response, chat_messages)

async def process_claude_response(response, chat_messages):
    """Process Claude's response, handling tool approvals"""
    while response.stop_reason == "tool_use":
        tool_use = next(block for block in response.content if block.type == "tool_use")
        
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

            messages = [
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": str(tool_result),
                        }
                    ],
                },
            ]

            chat_messages.extend(messages)
            response = await call_claude(chat_messages)

    final_response = next(
        (block.text for block in response.content if hasattr(block, "text")),
        None,
    )

    chat_messages = cl.user_session.get("chat_messages")
    chat_messages.append({"role": "assistant", "content": final_response})