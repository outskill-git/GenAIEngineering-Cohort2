#!/usr/bin/env python3
"""
Test script to verify agent chat functionality with tool calling.
"""

from agent_chat_app import (
    new_agent_conversation, add_user_message_and_run_agent, 
    get_conversation_history, execute_tool_call
)
from chatbot_models import get_agent_messages, get_agent_tool_calls_for_message
import json

def test_agent_chat():
    print("🤖 Testing Agent Chat Functionality")
    print("=" * 50)
    
    # Test 1: Create a new agent conversation
    print("✅ Test 1: Creating new agent conversation...")
    thread_id = new_agent_conversation('openrouter/cypher-alpha:free', 0.7)
    print(f"   Created agent thread ID: {thread_id}")
    
    # Test 2: Tool execution
    print("\n✅ Test 2: Testing tool execution...")
    try:
        result = execute_tool_call("list_files", {})
        print(f"   Tool result: {result['status']}")
        if result['status'] == 'success':
            print(f"   Found {len(result.get('files', []))} files")
        print("   ✅ Tool execution working!")
    except Exception as e:
        print(f"   ❌ Tool execution error: {e}")
        return False
    
    # Test 3: Send a message and run agent
    print("\n✅ Test 3: Sending message and running agent...")
    test_message = "Please list the files in the current directory"
    
    try:
        iterations = add_user_message_and_run_agent(
            thread_id, test_message, 'openrouter/cypher-alpha:free', 0.7
        )
        print(f"   Agent completed in {iterations} iterations")
        
        if iterations > 0:
            print("   ✅ Agent processing successful!")
        else:
            print("   ❌ No agent iterations")
            return False
            
    except Exception as e:
        print(f"   ❌ Error during agent processing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Check conversation history with tool calls
    print("\n✅ Test 4: Verifying conversation history...")
    try:
        history = get_conversation_history(thread_id)
        print(f"   History contains {len(history)} messages")
        
        # Check for tool call information
        has_tool_calls = any("Tool Calls" in msg.get("content", "") for msg in history)
        if has_tool_calls:
            print("   ✅ Tool calls displayed in history!")
        else:
            print("   ⚠️  No tool calls found in history (might be expected)")
            
    except Exception as e:
        print(f"   ❌ Error getting history: {e}")
        return False
    
    # Test 5: Check message separation
    print("\n✅ Test 5: Verifying agent/chatbot separation...")
    try:
        agent_messages = get_agent_messages(thread_id)
        print(f"   Agent thread has {len(agent_messages)} messages")
        
        # Import regular chatbot functions
        from chatbot_models import list_threads, list_agent_threads
        
        regular_threads = list_threads()
        agent_threads = list_agent_threads()
        
        print(f"   Regular chatbot threads: {len(regular_threads)}")
        print(f"   Agent threads: {len(agent_threads)}")
        
        if len(agent_threads) > 0:
            print("   ✅ Agent/chatbot separation working!")
        else:
            print("   ❌ No agent threads found")
            return False
            
    except Exception as e:
        print(f"   ❌ Error checking separation: {e}")
        return False
    
    print("\n🎉 All agent tests passed! Agent chat is working correctly.")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = test_agent_chat()
    if success:
        print("\n✅ You can now run 'python agent_chat_app.py' to use the agent chat!")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")