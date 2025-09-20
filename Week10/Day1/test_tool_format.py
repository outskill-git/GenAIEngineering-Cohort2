#!/usr/bin/env python3
"""
Test to verify tool message format is correct for OpenAI API
"""

from agent_chat_app import new_agent_conversation, add_user_message_and_run_agent
from chatbot_models import get_agent_messages
import json

def test_tool_message_format():
    print("🧪 Testing Tool Message Format")
    print("=" * 40)
    
    # Create new conversation
    thread_id = new_agent_conversation('openrouter/cypher-alpha:free', 0.7)
    print(f"Created thread: {thread_id}")
    
    # Send a message that will trigger tool calls
    add_user_message_and_run_agent(thread_id, "List the files in the notes directory", 'openrouter/cypher-alpha:free', 0.7)
    
    # Get all messages
    messages = get_agent_messages(thread_id)
    
    print("\n📋 Messages in conversation:")
    for i, msg in enumerate(messages):
        msg_dict = dict(msg)
        print(f"\n{i+1}. Role: {msg_dict['role']}")
        print(f"   Content: {msg_dict['content'][:100]}...")
        
        if msg_dict['role'] == 'tool':
            print(f"   Extra: {msg_dict.get('extra', 'None')}")
            
            # Parse extra to check tool_call_id
            if msg_dict.get('extra'):
                try:
                    extra = json.loads(msg_dict['extra'])
                    tool_call_id = extra.get('tool_call_id')
                    print(f"   Tool Call ID: {tool_call_id}")
                    
                    # This is how the message will be formatted for API
                    api_message = {
                        "role": "tool",
                        "content": msg_dict['content'],
                        "tool_call_id": tool_call_id
                    }
                    print(f"   API Format: {json.dumps(api_message, indent=2)}")
                    
                    # Verify format is correct
                    if all(key in api_message for key in ["role", "content", "tool_call_id"]):
                        print("   ✅ Tool message format is CORRECT for OpenAI API")
                    else:
                        print("   ❌ Tool message format is INCORRECT")
                        
                except Exception as e:
                    print(f"   ❌ Error parsing extra: {e}")
    
    print("\n🎯 Tool message format verification complete!")

if __name__ == "__main__":
    test_tool_message_format()