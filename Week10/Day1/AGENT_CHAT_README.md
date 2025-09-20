# Agent Chat with Tool Calling

This is an autonomous agent chat application that can use tools to interact with files and perform various tasks.

## Features

- **Multi-turn conversation** with persistent storage
- **Tool calling** capabilities with the following tools:
  - `list_files()` - Lists all files in the current directory
  - `read_file(filename)` - Reads content from a file
  - `create_file(name, content)` - Creates a new file with given content
  - `update_file(name, content)` - Updates an existing file with new content
- **Autonomous operation** - The agent continues to use tools until the task is complete
- **Conversation history** - All conversations are stored in SQLite database
- **Multiple model support** - Works with OpenAI, Anthropic, and other models via OpenRouter

## System Architecture

### Tool Calling Flow

1. **User Input**: User sends a message to the agent
2. **LLM Processing**: The LLM processes the message and decides which tools to call
3. **Tool Execution**: Tools are executed automatically and results are stored
4. **Tool Response**: Tool results are added as tool messages to the conversation
5. **Iteration**: The process continues until the LLM doesn't need to call any more tools

### Database Schema

The application uses SQLite with the following key tables:

- `threads` - Conversation threads
- `messages` - Individual messages in conversations
- `tool_calls` - Tool call tracking with status and results

### Tool Call Processing

```python
# Tool calls are detected in assistant messages
tool_calls = assistant_message.get("tool_calls", [])

# Each tool call is:
# 1. Stored in database with "pending" status
# 2. Executed using the tool function
# 3. Status updated to "completed" or "failed"
# 4. Result added as a tool message
```

## Usage

### Starting the Application

```bash
python3 agent_chat_app.py
```

The application will start a Gradio interface accessible at `http://localhost:7860`

### Using the Agent

1. **Start New Conversation**: Click "New Conversation" to begin
2. **Send Message**: Type your request and press Enter or click Send
3. **Watch Agent Work**: The agent will automatically use tools as needed
4. **View Results**: See the conversation history including tool calls and results

### Example Interactions

**Example 1: File Management**
```
User: "List all Python files in the current directory and show me the content of the main application file"

Agent will:
1. Call list_files() to get all files
2. Identify Python files from the list
3. Call read_file() on the main application file
4. Present the results to the user
```

**Example 2: Content Creation**
```
User: "Create a simple Python script that prints 'Hello, World!' and save it as hello.py"

Agent will:
1. Call create_file() with name="hello.py" and appropriate content
2. Confirm the file was created successfully
```

## Configuration

### Environment Variables

Create a `.env` file with your API keys:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
# OR
OPENAI_API_KEY=your_openai_api_key_here
```

### Model Selection

The application supports multiple models:
- OpenAI GPT-4o
- OpenAI GPT-4o Mini
- Anthropic Claude 3.5 Sonnet
- OpenRouter Cypher Alpha (Free)

## System Prompt

The agent uses the following system prompt:

```
"You are a helpful assistant that takes notes, maintains them and refines them! You are an autonomous agent, do not wait for user input, just use your tools to do the work."
```

## Technical Details

### Tool Definitions

Tools are defined using OpenAI's function calling format:

```python
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in the current directory",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    # ... other tools
]
```

### Error Handling

- Tool execution errors are captured and returned as error messages
- Failed tool calls are marked with "failed" status in the database
- The agent can handle and respond to tool failures gracefully

### Conversation Loop

The agent continues processing until:
1. The LLM doesn't call any tools in its response
2. Maximum iteration limit is reached (default: 10)
3. An error occurs that prevents further processing

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- `gradio` - Web interface
- `openai` - OpenAI API client
- `requests` - HTTP requests
- `sqlite3` - Database (built-in)
- `pydantic` - Data validation

## Testing

Run the test suite:

```bash
python3 test_agent.py
```

This will test:
- Individual tool functions
- Conversation flow
- Database operations

## Limitations

- Tools operate only in the current directory
- No file system navigation outside current directory
- Tool execution is synchronous
- No support for binary files (text files only)

## Future Enhancements

- Add more tools (web search, API calls, etc.)
- Support for file system navigation
- Async tool execution
- Tool result caching
- Custom tool definitions via configuration 