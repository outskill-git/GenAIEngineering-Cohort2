import gradio as gr
import os
import json
import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime
from chatbot_models import (
    init_db, insert_agent_thread, insert_agent_message, get_agent_messages, 
    get_agent_thread, list_agent_threads, insert_agent_tool_call, 
    update_agent_tool_call_status, get_pending_agent_tool_calls, 
    get_agent_tool_calls_for_message
)
from chatbot_openrouter import chat_openrouter
from agent_system_prompt import prompt as SYSTEM_PROMPT

# Initialize database
init_db()

# Tool functions
def list_files():
    """List all files in the notes/ directory"""
    try:
        # Ensure notes directory exists
        os.makedirs('notes', exist_ok=True)
        
        files = []
        for item in os.listdir('notes'):
            file_path = os.path.join('notes', item)
            if os.path.isfile(file_path):
                files.append(item)
        return {
            "status": "success",
            "files": files,
            "message": f"Found {len(files)} files in notes/ directory"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error listing files: {str(e)}"
        }

def read_file(filename: str):
    """Read content from a file in the notes/ directory"""
    try:
        # Ensure notes directory exists
        os.makedirs('notes', exist_ok=True)
        
        # Construct full path in notes directory
        file_path = os.path.join('notes', filename)
        
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "message": f"File '{filename}' not found in notes/ directory"
            }
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "status": "success",
            "content": content,
            "message": f"Successfully read {filename} from notes/ directory"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error reading file '{filename}': {str(e)}"
        }

def create_file(name: str, content: str):
    """Create a new file with given content in the notes/ directory"""
    try:
        # Ensure notes directory exists
        os.makedirs('notes', exist_ok=True)
        
        # Construct full path in notes directory
        file_path = os.path.join('notes', name)
        
        if os.path.exists(file_path):
            return {
                "status": "error",
                "message": f"File '{name}' already exists in notes/ directory"
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "status": "success",
            "message": f"Created successfully {name} in notes/ directory"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error creating file '{name}': {str(e)}"
        }

def update_file(name: str, content: str):
    """Update an existing file with new content in the notes/ directory"""
    try:
        # Ensure notes directory exists
        os.makedirs('notes', exist_ok=True)
        
        # Construct full path in notes directory
        file_path = os.path.join('notes', name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "status": "success",
            "message": f"Updated successfully {name} in notes/ directory"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error updating file '{name}': {str(e)}"
        }

# Tool registry
TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "create_file": create_file,
    "update_file": update_file
}

# Tool definitions for OpenAI API
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in the notes/ directory",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read content from a file in the notes/ directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to read"
                    }
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file with given content in the notes/ directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the file to create"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["name", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_file",
            "description": "Update an existing file with new content in the notes/ directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the file to update"
                    },
                    "content": {
                        "type": "string",
                        "description": "The new content for the file"
                    }
                },
                "required": ["name", "content"]
            }
        }
    }
]

def execute_tool_call(tool_name: str, arguments: dict):
    """Execute a tool call and return the result"""
    if tool_name not in TOOLS:
        return {
            "status": "error",
            "message": f"Unknown tool: {tool_name}"
        }
    
    try:
        tool_func = TOOLS[tool_name]
        if arguments:
            result = tool_func(**arguments)
        else:
            result = tool_func()
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error executing {tool_name}: {str(e)}"
        }

def process_tool_calls(thread_id: int, message_id: int, tool_calls: List[dict]) -> bool:
    """Process tool calls from an assistant message and return True when all are complete"""
    
    for tool_call in tool_calls:
        tool_call_id = tool_call.get("id")
        tool_name = tool_call.get("function", {}).get("name")
        arguments_str = tool_call.get("function", {}).get("arguments", "{}")
        
        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError:
            arguments = {}
        
        # Check if this tool call already exists in database
        existing_calls = get_agent_tool_calls_for_message(thread_id, message_id)
        already_exists = any(call["tool_call_id"] == tool_call_id for call in existing_calls)
        
        if not already_exists:
            # Store tool call in database BEFORE execution
            insert_agent_tool_call(
                thread_id=thread_id,
                message_id=message_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=arguments_str,
                status="pending"
            )
    
    # Now execute all pending tool calls for this message
    pending_calls = get_pending_agent_tool_calls(thread_id, message_id)
    
    for pending_call in pending_calls:
        tool_call_id = pending_call["tool_call_id"]
        tool_name = pending_call["tool_name"]
        arguments_str = pending_call["arguments"]
        
        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError:
            arguments = {}
        
        # Execute the tool call
        result = execute_tool_call(tool_name, arguments)
        
        # Update tool call status
        status = "completed" if result.get("status") == "success" else "failed"
        update_agent_tool_call_status(
            tool_call_id=tool_call_id,
            status=status,
            result=json.dumps(result)
        )
        
        # Check if we already have a tool message for this tool_call_id
        existing_messages = get_agent_messages(thread_id)
        tool_message_exists = False
        for msg_row in existing_messages:
            msg = dict(msg_row)  # Convert sqlite3.Row to dict
            if msg["role"] == "tool" and msg.get("extra"):
                try:
                    extra = json.loads(msg["extra"])
                    if extra.get("tool_call_id") == tool_call_id:
                        tool_message_exists = True
                        break
                except:
                    pass
        
        # Only add tool message if one doesn't already exist for this tool_call_id
        if not tool_message_exists:
            # Format tool content according to OpenAI standards
            if result.get("status") == "success":
                if "content" in result:
                    tool_content = result["content"]
                elif "files" in result:
                    tool_content = f"Found {len(result['files'])} files: {', '.join(result['files'])}"
                else:
                    tool_content = result.get("message", "Operation completed successfully")
            else:
                tool_content = result.get("message", "Operation failed")
            
            insert_agent_message(
                thread_id=thread_id,
                role="tool",
                content=tool_content,
                extra=json.dumps({
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "full_result": result
                })
            )
    
    # Check if all tool calls for this message are completed
    remaining_pending = get_pending_agent_tool_calls(thread_id, message_id)
    return len(remaining_pending) == 0

def agent_conversation_loop(thread_id: int, model: str = "openai/gpt-4o", temperature: float = 1.0, max_iterations: int = 10):
    """Run the agent conversation loop until no more tool calls are needed"""
    iterations = 0
    
    while iterations < max_iterations:
        iterations += 1
        
        # Get all messages for the thread
        msgs = get_agent_messages(thread_id)
        messages = []
        
        for m_row in msgs:
            m = dict(m_row)
            msg = {"role": m["role"], "content": m["content"]}
            
            # Add tool_call_id for tool messages
            if m["role"] == "tool" and m.get("extra"):
                try:
                    extra = json.loads(m["extra"])
                    if "tool_call_id" in extra:
                        msg["tool_call_id"] = extra["tool_call_id"]
                except:
                    pass
            
            # Add tool_calls for assistant messages
            elif m["role"] == "assistant" and m.get("extra"):
                try:
                    extra = json.loads(m["extra"])
                    # Check if this is the full API response data
                    if "choices" in extra:
                        assistant_message = extra.get("choices", [{}])[0].get("message", {})
                        tool_calls = assistant_message.get("tool_calls", [])
                        if tool_calls:
                            msg["tool_calls"] = tool_calls
                except:
                    pass
            
            messages.append(msg)
        
        # Make API call with tools
        assistant_content, data = chat_openrouter(
            messages=messages,
            model=model,
            temperature=temperature,
            tools=TOOL_DEFINITIONS
        )
        
        if data and "error" in data:
            # Add error message and break
            insert_agent_message(thread_id, "assistant", f"Error: {data['error']}")
            break
        
        # Extract assistant message and tool calls
        assistant_message = data.get("choices", [{}])[0].get("message", {})
        tool_calls = assistant_message.get("tool_calls", [])
        
        # Add assistant message (even if content is empty but has tool calls)
        message_id = insert_agent_message(
            thread_id=thread_id,
            role="assistant",
            content=assistant_content or "",
            extra=json.dumps(data) if data else None
        )
        
        # If no tool calls, we're done
        if not tool_calls:
            break
        
        # Process tool calls and wait for ALL to complete
        all_completed = process_tool_calls(thread_id, message_id, tool_calls)
        
        # Only continue if all tool calls are completed
        if not all_completed:
            print(f"Warning: Not all tool calls completed in iteration {iterations}")
            break
    
    return iterations

def new_agent_conversation(model: str = "openai/gpt-4o", temperature: float = 1.0) -> int:
    """Start a new agent conversation"""
    thread_id = insert_agent_thread()
    insert_agent_message(thread_id, "system", SYSTEM_PROMPT, model=model, temperature=temperature)
    return thread_id

def add_user_message_and_run_agent(thread_id: int, content: str, model: str = "openai/gpt-4o", temperature: float = 1.0):
    """Add user message and run the agent conversation loop"""
    # Add user message
    insert_agent_message(thread_id, "user", content, model=model, temperature=temperature)
    
    # Run agent loop
    iterations = agent_conversation_loop(thread_id, model, temperature)
    
    return iterations

def get_conversation_history(thread_id: int) -> List[dict]:
    """Get conversation history for display with enhanced tool call information"""
    msgs = get_agent_messages(thread_id)
    messages = []
    
    for m_row in msgs:
        m = dict(m_row)
        if m["role"] == "system":
            continue
        
        msg = {"role": m["role"], "content": m["content"]}
        
        # Enhanced handling for assistant messages
        if m["role"] == "assistant":
            # Get tool calls for this message
            tool_calls = get_agent_tool_calls_for_message(thread_id, m["id"])
            
            if tool_calls:
                # Show "Tool Calls" even if content is empty
                tool_call_summary = f"🛠️ Tool Calls ({len(tool_calls)}):\n"
                for tc in tool_calls:
                    status_emoji = "✅" if tc["status"] == "completed" else "❌" if tc["status"] == "failed" else "⏳"
                    tool_call_summary += f"- {status_emoji} {tc['tool_name']}\n"
                
                if msg["content"]:
                    msg["content"] = f"{msg['content']}\n\n{tool_call_summary}"
                else:
                    msg["content"] = tool_call_summary
            
            # Add metadata
            meta = {}
            for k in ["model", "temperature", "extra"]:
                if m.get(k) is not None:
                    meta[k] = m[k]
            if meta:
                msg["metadata"] = meta
        
        messages.append(msg)
    
    return messages

def build_agent_ui():
    """Build the Gradio UI for the agent chat"""
    from chatbot_models import get_default_title
    
    def get_sidebar_conversations(offset=0, max_display=10):
        threads = list_agent_threads()
        threads = sorted(threads, key=lambda t: t["updated_at"], reverse=True)
        total = len(threads)
        threads = threads[offset:offset+max_display]
        
        options = []
        thread_ids = []
        for t in threads:
            title = t["title"] or get_default_title(t["created_at"])
            options.append([f"🤖 {title}"])
            thread_ids.append(t["id"])
        
        more = (offset + max_display) < total
        return options, thread_ids, more, total
    
    def refresh_conversations():
        options, thread_ids, more, total = get_sidebar_conversations()
        return gr.update(value=options), thread_ids, gr.update(visible=more)
    
    def on_new_conversation(model, temperature):
        thread_id = new_agent_conversation(model, temperature)
        return thread_id, [], f"New agent conversation started (ID: {thread_id})"
    
    def on_send_message(message, thread_id, model, temperature):
        if not thread_id:
            return [], "Please start a new conversation first"
        
        if not message.strip():
            return get_conversation_history(thread_id), ""
        
        # Add user message and run agent
        iterations = add_user_message_and_run_agent(
            int(thread_id), message, model, temperature
        )
        
        # Get updated history
        history = get_conversation_history(int(thread_id))
        
        return history, f"Agent completed in {iterations} iterations"
    
    def on_select_conversation(evt: gr.SelectData, thread_ids):
        if not thread_ids or evt is None or evt.index is None or evt.index[0] >= len(thread_ids):
            return [], None, "No conversation selected"
        
        idx = evt.index[0]
        thread_id = thread_ids[idx]
        
        history = get_conversation_history(int(thread_id))
        return history, thread_id, f"Loaded agent conversation {thread_id}"
    
    def on_load_more(current_offset, current_thread_ids):
        new_offset = current_offset + 10
        options, thread_ids, more, total = get_sidebar_conversations(new_offset)
        return gr.update(value=options), thread_ids, gr.update(visible=more), new_offset
    
    with gr.Blocks(title="Agent Chat with Tools") as demo:
        gr.Markdown("# 🤖 Agent Chat with Tool Calling")
        gr.Markdown("This agent can use tools to list files, read files, create files, and update files in the notes/ folder. Tool calls are tracked and displayed even when message content is empty.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Agent Conversations")
                
                new_conv_btn = gr.Button("🆕 New Agent Conversation", variant="primary")
                
                conversation_list = gr.Dataframe(
                    headers=["Conversations"],
                    datatype=["str"],
                    interactive=True,
                    row_count=10,
                    col_count=1,
                    label="Load Agent Conversation"
                )
                
                load_more_btn = gr.Button("Load More", visible=False)
                refresh_btn = gr.Button("🔄 Refresh")
                
            with gr.Column(scale=3):
                gr.Markdown("### Agent Chat")
                
                chatbot = gr.Chatbot(
                    type="messages",
                    height=400,
                    show_label=False
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Type your message here...",
                        scale=4,
                        show_label=False
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                # Settings moved below message input
                with gr.Accordion("⚙️ Settings", open=False):
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            label="Model",
                            choices=[
                                "openrouter/cypher-alpha:free",
                                "openai/gpt-4o",
                                "openai/gpt-4.1",
                                "openai/gpt-4.1-mini",
                                "openai/gpt-4.1-nano",
                                "anthropic/claude-sonnet-4",
                                "anthropic/claude-3.5-sonnet"
                            ],
                            value="openrouter/cypher-alpha:free",
                            scale=2
                        )
                        temperature_slider = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            scale=1
                        )
                
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    max_lines=2
                )
        
        # State
        current_thread_id = gr.State(value=None)
        sidebar_thread_ids = gr.State(value=[])
        sidebar_offset = gr.State(value=0)
        
        # Event handlers
        new_conv_btn.click(
            fn=on_new_conversation,
            inputs=[model_dropdown, temperature_slider],
            outputs=[current_thread_id, chatbot, status_text]
        )
        
        send_btn.click(
            fn=on_send_message,
            inputs=[msg_input, current_thread_id, model_dropdown, temperature_slider],
            outputs=[chatbot, status_text]
        ).then(
            fn=lambda: "",
            outputs=[msg_input]
        )
        
        msg_input.submit(
            fn=on_send_message,
            inputs=[msg_input, current_thread_id, model_dropdown, temperature_slider],
            outputs=[chatbot, status_text]
        ).then(
            fn=lambda: "",
            outputs=[msg_input]
        )
        
        refresh_btn.click(
            fn=refresh_conversations,
            outputs=[conversation_list, sidebar_thread_ids, load_more_btn]
        )
        
        conversation_list.select(
            fn=on_select_conversation,
            inputs=[sidebar_thread_ids],
            outputs=[chatbot, current_thread_id, status_text]
        )
        
        load_more_btn.click(
            fn=on_load_more,
            inputs=[sidebar_offset, sidebar_thread_ids],
            outputs=[conversation_list, sidebar_thread_ids, load_more_btn, sidebar_offset]
        )
        
        # Initialize sidebar on load
        demo.load(
            fn=refresh_conversations,
            outputs=[conversation_list, sidebar_thread_ids, load_more_btn]
        )
    
    return demo

if __name__ == "__main__":
    demo = build_agent_ui()
    demo.launch()