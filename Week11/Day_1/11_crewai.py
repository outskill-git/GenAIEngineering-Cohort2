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

# Define Callback Functions
def my_step_callback(step):
    """Called after each step completes"""
    print(f"🔄 Step completed: {step}")

def my_task_callback(output):
    """Called after each task completes"""
    print(f"✅ Task finished! Output preview: {output.raw[:100]}...")

# Create Agent
writer = Agent(
    role='Content Writer',
    goal='Write short content pieces',
    backstory='You are a fast content writer.',
    llm=llm
)

# Create Tasks
task1 = Task(
    description='Write a title for an article about AI',
    expected_output='A catchy title',
    agent=writer
)

task2 = Task(
    description='Write a short intro paragraph about AI',
    expected_output='An engaging intro paragraph',
    agent=writer
)

# Create Crew with Callbacks
crew = Crew(
    agents=[writer],
    tasks=[task1, task2],
    step_callback=my_step_callback,    # Called after each step
    task_callback=my_task_callback     # Called after each task
)

# Run with Callbacks
print("🤖 Starting Callbacks Demo...")
result = crew.kickoff()
print("🎉 All done!", result)