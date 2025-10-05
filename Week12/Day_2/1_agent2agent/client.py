"""
talk_to_agents.py - Fixed version
"""
import requests, uuid, time, sys

class A2AClient:
    def __init__(self, url, name):
        self.url, self.name = url, name

    def send(self, text):
        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": text}],
                    "kind": "message",
                    "messageId": str(uuid.uuid4())
                }
            },
            "id": 1
        }
        response = requests.post(self.url, json=payload)
        return response.json()['result']['id']

    def get_result(self, task_id):
        for i in range(30):
            payload = {
                "jsonrpc": "2.0",
                "method": "tasks/get",
                "params": {"id": task_id},
                "id": 2
            }
            result = requests.post(self.url, json=payload).json()['result']

            # FIXED: state is inside status object
            state = result['status']['state']

            if state == 'completed':
                # Get the agent's response from history
                for msg in reversed(result['history']):
                    if msg['role'] == 'agent':
                        return msg['parts'][0]['text']
                return "No response found"
            elif state == 'failed':
                return "Task failed"

            if i % 5 == 0:
                print(f"  [{state}]")
            time.sleep(1)
        return "Timeout"

    def ask(self, q):
        print(f"\n{'='*60}")
        print(f"{self.name}: {q}")
        print('='*60)
        task_id = self.send(q)
        answer = self.get_result(task_id)
        print(f"\n{answer}\n{'='*60}")
        return answer

# Check agents
try:
    requests.get("http://localhost:9321/.well-known/agent.json", timeout=2)
    print("✅ CrewAI running")
except:
    print("❌ Start: python crewai_agent.py")
    sys.exit(1)

try:
    requests.get("http://localhost:9331/.well-known/agent.json", timeout=2)
    print("✅ Phidata running")
except:
    print("❌ Start: python phidata_agent.py")
    sys.exit(1)

print("\n" + "="*60)
print("Agent-to-Agent Communication Demo")
print("="*60)

crewai = A2AClient("http://localhost:9321", "CrewAI")
phidata = A2AClient("http://localhost:9331", "Phidata")

print("\n📍 DEMO 1: Simple Communication")
crewai.ask("What are the top 3 AI trends in 2025?")
phidata.ask("What are the best tech stocks to invest in?")

print("\n📍 DEMO 2: Agent Collaboration")
research = crewai.ask("Give a brief overview of the AI market")
if research and len(research) > 100:
    phidata.ask(f"Based on this AI market research:\n\n{research[:500]}...\n\nProvide investment recommendations")

print("\n✅ All demos completed!")