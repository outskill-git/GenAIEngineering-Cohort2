#!/usr/bin/env python3
"""
Test script to verify chatbot functionality is working correctly.
"""

from chatbot_app import new_conversation, gradio_stream, get_history
from chatbot_models import get_messages
import json

def test_chatbot():
    print("🧪 Testing Chatbot Functionality")
    print("=" * 50)
    
    # Test 1: Create a new conversation
    print("✅ Test 1: Creating new conversation...")
    tid = new_conversation('openrouter/cypher-alpha:free', 0.7)
    print(f"   Created thread ID: {tid}")
    
    # Test 2: Send a message and get response
    print("\n✅ Test 2: Sending message and getting response...")
    test_message = "Hello! Please respond with exactly 'Test successful' if you can understand me."
    
    try:
        response_count = 0
        for result in gradio_stream(test_message, tid, 'openrouter/cypher-alpha:free', 0.7, None, None):
            response_count += 1
            if result and len(result) >= 2:
                user_msg = result[-2]
                assistant_msg = result[-1]
                print(f"   User: {user_msg['content']}")
                print(f"   Assistant: {assistant_msg['content']}")
                break
        
        if response_count > 0:
            print("   ✅ Message exchange successful!")
        else:
            print("   ❌ No response received")
            
    except Exception as e:
        print(f"   ❌ Error during message exchange: {e}")
        return False
    
    # Test 3: Verify message persistence
    print("\n✅ Test 3: Verifying message persistence...")
    try:
        messages = get_messages(tid)
        if len(messages) >= 2:
            print(f"   Found {len(messages)} messages in database")
            print("   ✅ Message persistence working!")
        else:
            print(f"   ❌ Expected at least 2 messages, found {len(messages)}")
            return False
    except Exception as e:
        print(f"   ❌ Error checking persistence: {e}")
        return False
    
    # Test 4: Test conversation history
    print("\n✅ Test 4: Testing conversation history...")
    try:
        history = get_history(tid)
        if len(history) >= 2:
            print(f"   History contains {len(history)} messages")
            print("   ✅ History retrieval working!")
        else:
            print(f"   ❌ Expected at least 2 messages in history, found {len(history)}")
            return False
    except Exception as e:
        print(f"   ❌ Error getting history: {e}")
        return False
    
    print("\n🎉 All tests passed! Chatbot is working correctly.")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = test_chatbot()
    if success:
        print("\n✅ You can now run 'python chatbot_app.py' to use the chatbot!")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")