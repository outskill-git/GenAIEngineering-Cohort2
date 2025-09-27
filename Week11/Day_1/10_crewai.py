import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'
os.environ['OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'

llm = LLM(
        model='openai/gpt-4o',
        api_key=os.getenv('OPEN_ROUTER_KEY'),
        base_url="https://openrouter.ai/api/v1"
    )

# Create Agent
writer = Agent(
    role='Content Writer',
    goal='Write articles with human feedback',
    backstory='You are a writer who values human input.',
    llm=llm
)

# Create Task with Human Input
writing_task = Task(
    description='Write a short article about AI. Ask human for feedback before finalizing.',
    expected_output='A polished article approved by human.',
    agent=writer,
    human_input=True  # This is the key! Enables human-in-the-loop
)

# Create Crew
crew = Crew(
    agents=[writer],
    tasks=[writing_task]
)

# Run with Human Interaction
print("🤖 Starting Human-in-the-Loop Demo...")
result = crew.kickoff()
print("✅ Final Result:", result)