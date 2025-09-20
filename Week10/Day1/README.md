# GenAI Engineering - Week 10 Chatbot Apps

## Setup

### 1. Install Dependencies
`pip install argparse deepeval fastapi langfuse openai openai-agents python-dotenv uvicorn gradio pydantic requests`

OR

```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Create `.env` file:
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 3. Initialize Database
```bash
python scripts/seed_db.py
```

## Apps

### 1. Multi-turn Chatbot
```bash
python chatbot_app.py
```
- Conversation history with sidebar
- Multiple model support
- JSON response viewer

### 2. Quiz Builder
```bash
python quiz_app.py
```
- Generate 5-question multiple choice quizzes
- Structured LLM responses
- Score tracking and history

### 3. Agent Chat with Tools
```bash
python agent_chat_app.py
```
- File operations in `notes/` folder
- Tool calling with OpenAI API compliance
- Conversation tracking separate from chatbot

## Testing
```bash
python test_chatbot.py    # Test chatbot functionality
python test_agent_app.py  # Test agent functionality
```