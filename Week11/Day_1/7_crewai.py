
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM, Process
from crewai_tools import SerperDevTool, FileReadTool


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'
os.environ['OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'


search_tool = SerperDevTool()
file_tool = FileReadTool()

startup_idea = "AI-driven personalized learning platform"

llm = LLM(
        model='openai/gpt-4o',
        api_key=os.getenv('OPEN_ROUTER_KEY'),
        base_url="https://openrouter.ai/api/v1"
    )
# Create Agents for Iterative Process
writer = Agent(
    role='Content Writer',
    goal='Write and improve content through iterations',
    backstory='You write content and improve it based on feedback on startup ventures in {startup_idea}',
    llm=llm
)

reviewer = Agent(
    role='Content Reviewer',
    goal='Review content and provide improvement suggestions',
    backstory='You review content and suggest specific improvements.',
    llm=llm
)

# ITERATION 1: First Draft
draft_task = Task(
    description='Write a first draft of an article about "Future of Remote Work"',
    expected_output='A basic article draft (300-400 words)',
    agent=writer
)

# ITERATION 2: Review & Feedback
review_task = Task(
    description='''Review the article draft and provide specific feedback:
    1. What needs improvement?
    2. What's missing?
    3. Specific suggestions for enhancement

    Focus on structure, clarity, and engagement.''',
    expected_output='Detailed feedback with specific improvement suggestions',
    agent=reviewer,
    context=[draft_task]  # Uses output from draft_task
)

# ITERATION 3: Improved Version
improve_task = Task(
    description='''Rewrite the article based on the reviewer's feedback.
    Address all suggested improvements and create a much better version.''',
    expected_output='An improved, polished article incorporating all feedback',
    agent=writer,
    context=[draft_task, review_task]  # Uses both previous outputs
)

# ITERATION 4: Final Review
final_review_task = Task(
    description='''Do a final review of the improved article:
    1. Is it significantly better than the first draft?
    2. Are all previous issues resolved?
    3. Any final polish needed?''',
    expected_output='Final assessment and any last-minute suggestions',
    agent=reviewer,
    context=[improve_task]  # Uses improved version
)

# ITERATION 5: Final Polish
final_task = Task(
    description='''Create the final polished version:
    - Incorporate any final suggestions
    - Ensure perfect grammar and flow
    - Add engaging title and conclusion''',
    expected_output='Publication-ready final article',
    agent=writer,
    context=[improve_task, final_review_task]  # Uses improved version + final feedback
)

# Create Iterative Crew
crew = Crew(
    agents=[writer, reviewer],
    tasks=[draft_task, review_task, improve_task, final_review_task, final_task],
    verbose=True
)



result = crew.kickoff()
print(result)
