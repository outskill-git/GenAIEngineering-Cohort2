from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import os
import json
import sqlite3

CONV_DIR = "conversations"
os.makedirs(CONV_DIR, exist_ok=True)

DB_PATH = "chatbot.sqlite3"

def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL;')
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS threads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id INTEGER,
        role TEXT,
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        model TEXT,
        temperature REAL,
        provider TEXT,
        object TEXT,
        created INTEGER,
        choices TEXT,
        usage_prompt_tokens INTEGER,
        usage_completion_tokens INTEGER,
        usage_total_tokens INTEGER,
        extra TEXT,
        FOREIGN KEY(thread_id) REFERENCES threads(id)
    )''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS app_defaults (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        default_chat_model TEXT,
        default_agent_model TEXT,
        default_temperature REAL,
        default_max_tokens INTEGER
    )''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS quizzes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        topic TEXT NOT NULL,
        raw_json TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        quiz_id INTEGER,
        question_text TEXT NOT NULL,
        option_1 TEXT NOT NULL,
        option_2 TEXT NOT NULL,
        option_3 TEXT NOT NULL,
        option_4 TEXT NOT NULL,
        correct_answer INTEGER NOT NULL,
        question_order INTEGER NOT NULL,
        FOREIGN KEY(quiz_id) REFERENCES quizzes(id)
    )''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS quiz_attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        quiz_id INTEGER,
        score INTEGER NOT NULL,
        total_questions INTEGER NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(quiz_id) REFERENCES quizzes(id)
    )''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS quiz_responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        attempt_id INTEGER,
        question_id INTEGER,
        selected_answer INTEGER NOT NULL,
        is_correct BOOLEAN NOT NULL,
        FOREIGN KEY(attempt_id) REFERENCES quiz_attempts(id),
        FOREIGN KEY(question_id) REFERENCES questions(id)
    )''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS tool_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id INTEGER,
        message_id INTEGER,
        tool_call_id TEXT,
        tool_name TEXT,
        arguments TEXT,
        status TEXT,
        result TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(thread_id) REFERENCES threads(id),
        FOREIGN KEY(message_id) REFERENCES messages(id)
    )''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS agent_threads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS agent_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id INTEGER,
        role TEXT,
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        model TEXT,
        temperature REAL,
        provider TEXT,
        object TEXT,
        created INTEGER,
        choices TEXT,
        usage_prompt_tokens INTEGER,
        usage_completion_tokens INTEGER,
        usage_total_tokens INTEGER,
        extra TEXT,
        FOREIGN KEY(thread_id) REFERENCES agent_threads(id)
    )''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS agent_tool_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id INTEGER,
        message_id INTEGER,
        tool_call_id TEXT,
        tool_name TEXT,
        arguments TEXT,
        status TEXT,
        result TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(thread_id) REFERENCES agent_threads(id),
        FOREIGN KEY(message_id) REFERENCES agent_messages(id)
    )''')
    conn.commit()
    conn.close()

class Message(BaseModel):
    role: str
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    extra: Optional[Dict[str, Any]] = None

class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    model: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

class QuizQuestion(BaseModel):
    id: int
    question: str
    options: List[str]
    correct_answer: int

class Quiz(BaseModel):
    quiz: str
    questions: List[QuizQuestion]

def save_conversation(conv: Conversation):
    conv.updated_at = datetime.utcnow().isoformat()
    with open(f"{CONV_DIR}/{conv.id}.json", "w") as f:
        f.write(json.dumps(conv.dict(), indent=2))

def load_conversation(conv_id: str) -> Conversation:
    with open(f"{CONV_DIR}/{conv_id}.json", "r") as f:
        return Conversation.parse_raw(f.read())

def list_conversations() -> List[Dict[str, str]]:
    files = [f for f in os.listdir(CONV_DIR) if f.endswith(".json")]
    convs = []
    for fname in files:
        with open(f"{CONV_DIR}/{fname}", "r") as f:
            data = json.load(f)
            convs.append({
                "id": data["id"],
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
                "model": data.get("model", "")
            })
    convs.sort(key=lambda x: x["updated_at"], reverse=True)
    return convs

def insert_thread(title=None):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO threads (title) VALUES (?)", (title,))
    thread_id = c.lastrowid
    conn.commit()
    conn.close()
    return thread_id

def update_thread_title(thread_id, title):
    conn = get_db()
    c = conn.cursor()
    c.execute("UPDATE threads SET title=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (title, thread_id))
    conn.commit()
    conn.close()

def get_thread(thread_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM threads WHERE id=?", (thread_id,))
    row = c.fetchone()
    conn.close()
    return row

def list_threads():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM threads ORDER BY updated_at DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def insert_message(thread_id, role, content, timestamp=None, model=None, temperature=None, provider=None, object_=None, created=None, choices=None, usage_prompt_tokens=None, usage_completion_tokens=None, usage_total_tokens=None, extra=None):
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO messages (thread_id, role, content, timestamp, model, temperature, provider, object, created, choices, usage_prompt_tokens, usage_completion_tokens, usage_total_tokens, extra)
        VALUES (?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (thread_id, role, content, timestamp, model, temperature, provider, object_, created, choices, usage_prompt_tokens, usage_completion_tokens, usage_total_tokens, extra))
    msg_id = c.lastrowid
    conn.commit()
    conn.close()
    return msg_id

def get_messages(thread_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM messages WHERE thread_id=? ORDER BY timestamp ASC, id ASC", (thread_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def get_default_title(created_at):
    dt = datetime.fromisoformat(created_at) if isinstance(created_at, str) else created_at
    return dt.strftime('%b %d %I:%M %p')

def get_app_defaults():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM app_defaults ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    return row

def set_app_defaults(default_chat_model=None, default_agent_model=None, default_temperature=None, default_max_tokens=None):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO app_defaults (default_chat_model, default_agent_model, default_temperature, default_max_tokens) VALUES (?, ?, ?, ?)", (default_chat_model, default_agent_model, default_temperature, default_max_tokens))
    conn.commit()
    conn.close()

def insert_quiz(title, topic, raw_json):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO quizzes (title, topic, raw_json) VALUES (?, ?, ?)", (title, topic, raw_json))
    quiz_id = c.lastrowid
    conn.commit()
    conn.close()
    return quiz_id

def insert_question(quiz_id, question_text, option_1, option_2, option_3, option_4, correct_answer, question_order):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO questions (quiz_id, question_text, option_1, option_2, option_3, option_4, correct_answer, question_order) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", 
              (quiz_id, question_text, option_1, option_2, option_3, option_4, correct_answer, question_order))
    question_id = c.lastrowid
    conn.commit()
    conn.close()
    return question_id

def get_quiz(quiz_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM quizzes WHERE id=?", (quiz_id,))
    row = c.fetchone()
    conn.close()
    return row

def get_quiz_questions(quiz_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM questions WHERE quiz_id=? ORDER BY question_order ASC", (quiz_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def list_quizzes():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM quizzes ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def insert_quiz_attempt(quiz_id, score, total_questions):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO quiz_attempts (quiz_id, score, total_questions) VALUES (?, ?, ?)", (quiz_id, score, total_questions))
    attempt_id = c.lastrowid
    conn.commit()
    conn.close()
    return attempt_id

def insert_quiz_response(attempt_id, question_id, selected_answer, is_correct):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO quiz_responses (attempt_id, question_id, selected_answer, is_correct) VALUES (?, ?, ?, ?)", 
              (attempt_id, question_id, selected_answer, is_correct))
    response_id = c.lastrowid
    conn.commit()
    conn.close()
    return response_id

def insert_tool_call(thread_id, message_id, tool_call_id, tool_name, arguments, status="pending", result=None):
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        INSERT INTO tool_calls (thread_id, message_id, tool_call_id, tool_name, arguments, status, result)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (thread_id, message_id, tool_call_id, tool_name, arguments, status, result))
    tool_call_row_id = c.lastrowid
    conn.commit()
    conn.close()
    return tool_call_row_id

def update_tool_call_status(tool_call_id, status, result=None):
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        UPDATE tool_calls SET status=?, result=?, updated_at=CURRENT_TIMESTAMP WHERE tool_call_id=?
    ''', (status, result, tool_call_id))
    conn.commit()
    conn.close()

def get_pending_tool_calls(thread_id, message_id):
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        SELECT * FROM tool_calls WHERE thread_id=? AND message_id=? AND status=?
    ''', (thread_id, message_id, "pending"))
    rows = c.fetchall()
    conn.close()
    return rows

def get_tool_calls_for_message(thread_id, message_id):
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        SELECT * FROM tool_calls WHERE thread_id=? AND message_id=?
    ''', (thread_id, message_id))
    rows = c.fetchall()
    conn.close()
    return rows

# Agent-specific database functions
def insert_agent_thread(title=None):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO agent_threads (title) VALUES (?)", (title,))
    thread_id = c.lastrowid
    conn.commit()
    conn.close()
    return thread_id

def get_agent_thread(thread_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM agent_threads WHERE id=?", (thread_id,))
    row = c.fetchone()
    conn.close()
    return row

def list_agent_threads():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM agent_threads ORDER BY updated_at DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def insert_agent_message(thread_id, role, content, timestamp=None, model=None, temperature=None, provider=None, object_=None, created=None, choices=None, usage_prompt_tokens=None, usage_completion_tokens=None, usage_total_tokens=None, extra=None):
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO agent_messages (thread_id, role, content, timestamp, model, temperature, provider, object, created, choices, usage_prompt_tokens, usage_completion_tokens, usage_total_tokens, extra)
        VALUES (?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (thread_id, role, content, timestamp, model, temperature, provider, object_, created, choices, usage_prompt_tokens, usage_completion_tokens, usage_total_tokens, extra))
    msg_id = c.lastrowid
    conn.commit()
    conn.close()
    return msg_id

def get_agent_messages(thread_id):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM agent_messages WHERE thread_id=? ORDER BY timestamp ASC, id ASC", (thread_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def insert_agent_tool_call(thread_id, message_id, tool_call_id, tool_name, arguments, status="pending", result=None):
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        INSERT INTO agent_tool_calls (thread_id, message_id, tool_call_id, tool_name, arguments, status, result)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (thread_id, message_id, tool_call_id, tool_name, arguments, status, result))
    tool_call_row_id = c.lastrowid
    conn.commit()
    conn.close()
    return tool_call_row_id

def update_agent_tool_call_status(tool_call_id, status, result=None):
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        UPDATE agent_tool_calls SET status=?, result=?, updated_at=CURRENT_TIMESTAMP WHERE tool_call_id=?
    ''', (status, result, tool_call_id))
    conn.commit()
    conn.close()

def get_agent_tool_calls_for_message(thread_id, message_id):
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        SELECT * FROM agent_tool_calls WHERE thread_id=? AND message_id=?
    ''', (thread_id, message_id))
    rows = c.fetchall()
    conn.close()
    return rows

def get_pending_agent_tool_calls(thread_id, message_id):
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        SELECT * FROM agent_tool_calls WHERE thread_id=? AND message_id=? AND status=?
    ''', (thread_id, message_id, "pending"))
    rows = c.fetchall()
    conn.close()
    return rows 