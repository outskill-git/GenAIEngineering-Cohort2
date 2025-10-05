"""
phidata_agent.py - Phidata A2A Agent
Run: python phidata_agent.py
"""
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from fasta2a import FastA2A, Worker
from fasta2a.broker import InMemoryBroker
from fasta2a.storage import InMemoryStorage
from fasta2a.schema import Message, TextPart, TaskSendParams, TaskIdParams
import os
from dotenv import load_dotenv
import uvicorn

load_dotenv()

import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

env_vars_to_clear = ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_API_BASE']
for var in env_vars_to_clear:
    if os.getenv(var):
        print(f"⚠️  Removing conflicting {var}")
        del os.environ[var]

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'
os.environ['OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'

Context = list[Message]

class PhidataWorker(Worker[Context]):
    def __init__(self, storage, broker):
        super().__init__(storage=storage, broker=broker)
        self.agent = Agent(name="Financial Analyst", model=OpenAIChat(id="gpt-3.5-turbo"), description="Expert", instructions=["Analyze"], markdown=False)

    async def run_task(self, params: TaskSendParams) -> None:
        task = await self.storage.load_task(params['id'])
        if not task: return
        await self.storage.update_task(task['id'], state='working')

        user_message = ""
        for msg in task.get('history', []):
            if msg.get('role') == 'user':
                for part in msg.get('parts', []):
                    if part.get('kind') == 'text':
                        user_message += part.get('text', '')

        try:
            response_obj = self.agent.run(user_message)
            result = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)

            response = Message(role='agent', parts=[TextPart(text=f"[Phidata]\n{result}", kind='text')], kind='message', message_id=str(uuid.uuid4()))
            context = await self.storage.load_context(task['context_id']) or []
            context.append(response)
            await self.storage.update_context(task['context_id'], context)
            await self.storage.update_task(task['id'], state='completed', new_messages=[response])
        except Exception as e:
            error_msg = Message(role='agent', parts=[TextPart(text=f"Error: {e}", kind='text')], kind='message', message_id=str(uuid.uuid4()))
            await self.storage.update_task(task['id'], state='failed', new_messages=[error_msg])

    async def cancel_task(self, params: TaskIdParams) -> None:
        await self.storage.update_task(params['id'], state='cancelled')

    def build_message_history(self, history): return history
    def build_artifacts(self, result): return []

storage, broker = InMemoryStorage(), InMemoryBroker()
worker = PhidataWorker(storage=storage, broker=broker)

@asynccontextmanager
async def lifespan(app: FastA2A) -> AsyncIterator[None]:
    async with app.task_manager:
        async with worker.run():
            yield

app = FastA2A(storage=storage, broker=broker, lifespan=lifespan)

if __name__ == "__main__":
    print("🟢 Phidata on http://localhost:9331")
    uvicorn.run(app, host="0.0.0.0", port=9331, log_level="warning")