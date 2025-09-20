import os
import sys
import glob
import json
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chatbot.sqlite3')
CONV_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'conversations')

SCHEMA_SQL = [
    '''CREATE TABLE IF NOT EXISTS threads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''',
    '''CREATE TABLE IF NOT EXISTS messages (
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
    )''',
    '''CREATE TABLE IF NOT EXISTS app_defaults (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        default_chat_model TEXT,
        default_agent_model TEXT,
        default_temperature REAL,
        default_max_tokens INTEGER
    )''',
    '''CREATE TABLE IF NOT EXISTS quizzes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        topic TEXT NOT NULL,
        raw_json TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''',
    '''CREATE TABLE IF NOT EXISTS questions (
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
    )''',
    '''CREATE TABLE IF NOT EXISTS quiz_attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        quiz_id INTEGER,
        score INTEGER NOT NULL,
        total_questions INTEGER NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(quiz_id) REFERENCES quizzes(id)
    )''',
    '''CREATE TABLE IF NOT EXISTS quiz_responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        attempt_id INTEGER,
        question_id INTEGER,
        selected_answer INTEGER NOT NULL,
        is_correct BOOLEAN NOT NULL,
        FOREIGN KEY(attempt_id) REFERENCES quiz_attempts(id),
        FOREIGN KEY(question_id) REFERENCES questions(id)
    )''',
    '''CREATE TABLE IF NOT EXISTS agent_threads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )''',
    '''CREATE TABLE IF NOT EXISTS agent_messages (
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
    )''',
    '''CREATE TABLE IF NOT EXISTS agent_tool_calls (
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
    )'''
]

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for stmt in SCHEMA_SQL:
        c.execute(stmt)
    conn.commit()
    conn.close()

print("Initializing DB...")
try:
    init_db()
    print("DB initialized.")
except Exception as e:
    print(f"DB init failed or already exists: {e}")

json_files = glob.glob(os.path.join(CONV_DIR, '*.json'))
print(f"Found {len(json_files)} conversation files.")

def insert_thread(title=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO threads (title) VALUES (?)", (title,))
    thread_id = c.lastrowid
    conn.commit()
    conn.close()
    return thread_id

def insert_message(thread_id, role, content, timestamp=None, model=None, temperature=None, provider=None, object_=None, created=None, choices=None, usage_prompt_tokens=None, usage_completion_tokens=None, usage_total_tokens=None, extra=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO messages (thread_id, role, content, timestamp, model, temperature, provider, object, created, choices, usage_prompt_tokens, usage_completion_tokens, usage_total_tokens, extra)
        VALUES (?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (thread_id, role, content, timestamp, model, temperature, provider, object_, created, choices, usage_prompt_tokens, usage_completion_tokens, usage_total_tokens, extra))
    msg_id = c.lastrowid
    conn.commit()
    conn.close()
    return msg_id

for jf in json_files:
    try:
        with open(jf, 'r') as f:
            data = json.load(f)
        title = data.get('title')
        created_at = data.get('created_at')
        thread_id = insert_thread(title=title)
        print(f"Inserted thread {thread_id} from {os.path.basename(jf)}")
        for msg in data.get('messages', []):
            extra = msg.get('extra') or {}
            usage = extra.get('usage', {})
            insert_message(
                thread_id=thread_id,
                role=msg.get('role'),
                content=msg.get('content'),
                timestamp=msg.get('timestamp'),
                model=data.get('model'),
                temperature=extra.get('temperature'),
                provider=extra.get('provider'),
                object_=extra.get('object'),
                created=extra.get('created'),
                choices=json.dumps(extra.get('choices')) if extra.get('choices') else None,
                usage_prompt_tokens=usage.get('prompt_tokens'),
                usage_completion_tokens=usage.get('completion_tokens'),
                usage_total_tokens=usage.get('total_tokens'),
                extra=json.dumps(extra) if extra else None
            )
        print(f"Inserted {len(data.get('messages', []))} messages for thread {thread_id}")
    except Exception as e:
        print(f"Error processing {jf}: {e}")
print("Seeding complete.") 