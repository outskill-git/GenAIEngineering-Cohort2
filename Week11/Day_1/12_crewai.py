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

# Create Agents with Different Skills
researcher = Agent(
    role='Research Specialist',
    goal='Find facts and data about topics',
    backstory='You are an expert at finding accurate information.',
    llm=llm,
    allow_delegation=False  # Cannot delegate to others
)

writer = Agent(
    role='Content Writer',
    goal='Create engaging content',
    backstory='You write great content but sometimes need research help.',
    llm=llm,
    allow_delegation=True   # Can ask others for help!
)

editor = Agent(
    role='Editor',
    goal='Review and improve content',
    backstory='You polish content and can delegate research or writing tasks.',
    llm=llm,
    allow_delegation=True   # Can delegate to both researcher and writer
)

# Create Task for Editor (who can delegate)
content_task = Task(
    description="""Create a complete blog post about "Benefits of AI in Healthcare".

    You may need to:
    1. Research current AI healthcare applications
    2. Write engaging content
    3. Edit and polish the final piece

    Feel free to delegate research or writing to appropriate team members.""",

    expected_output='A well-researched, engaging blog post about AI in healthcare',
    agent=editor  # Editor can delegate parts of this task
)

# Create Crew with Delegation Enabled
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[content_task],
    verbose=True  # See delegation in action!
)

# Run with Delegation
print("🤖 Starting Delegation Demo...")
print("👥 Editor can delegate to Researcher and Writer")
print("🔍 Researcher specializes in finding facts")
print("✍️ Writer specializes in creating content")
print("=" * 50)

result = crew.kickoff()
print("🎉 Collaboration complete!", result)