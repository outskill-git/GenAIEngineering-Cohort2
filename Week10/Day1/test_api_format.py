#!/usr/bin/env python3
"""
Test to verify API message format includes tool_calls in assistant messages
"""

from agent_chat_app import new_agent_conversation, agent_conversation_loop
from chatbot_models import get_agent_messages, insert_agent_message
import json

def test_api_message_format():
    print("🧪 Testing API Message Format with Tool Calls")
    print("=" * 50)
    
    # Create new conversation
    thread_id = new_agent_conversation('openrouter/cypher-alpha:free', 0.7)
    print(f"Created thread: {thread_id}")
    
    # Add user message
    insert_agent_message(thread_id, "user", "List files in the notes directory", model='openrouter/cypher-alpha:free', temperature=0.7)
    
    try:
        # Run one iteration of the agent loop
        iterations = agent_conversation_loop(thread_id, 'openrouter/cypher-alpha:free', 0.7, max_iterations=1)
        print(f"Agent completed {iterations} iterations")
        
        # Get all messages and show how they would be formatted for API
        messages = get_agent_messages(thread_id)
        
        print("\n📋 API Message Format:")
        api_messages = []
        
        for m_row in messages:
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
                            print(f"✅ Found tool_calls in assistant message: {len(tool_calls)} calls")
                except:
                    pass
            
            api_messages.append(msg)
        
        # Check message sequence
        for i, msg in enumerate(api_messages):
            if msg["role"] == "system":
                continue
                
            print(f"\n{i}. Role: {msg['role']}")
            print(f"   Content: {msg['content'][:100]}...")
            
            if msg['role'] == 'assistant' and 'tool_calls' in msg:
                print(f"   Tool Calls: {len(msg['tool_calls'])}")
                for j, tc in enumerate(msg['tool_calls']):
                    print(f"     {j+1}. {tc['function']['name']} (ID: {tc['id']})")
                    
            if msg['role'] == 'tool':
                print(f"   Tool Call ID: {msg.get('tool_call_id', 'MISSING!')}")
                
                # Verify this tool message has a preceding assistant with tool_calls
                for prev_i in range(i-1, -1, -1):
                    prev_msg = api_messages[prev_i]
                    if prev_msg['role'] == 'assistant' and 'tool_calls' in prev_msg:
                        # Check if this tool_call_id matches any in the assistant message
                        matching_call = any(tc['id'] == msg.get('tool_call_id') for tc in prev_msg['tool_calls'])
                        if matching_call:
                            print(f"   ✅ Tool message properly linked to assistant message {prev_i}")
                            break
                else:
                    print(f"   ❌ Tool message has no matching assistant message with tool_calls!")
        
        print(f"\n🎯 API message format test complete!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api_message_format()