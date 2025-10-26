
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM, Process
from crewai_tools import SerperDevTool


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'
os.environ['OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'

topic = "Artificial Intelligence in Healthcare"
# Define multiple agents for a content creation workflow
search_tool = SerperDevTool()
# 1. Content Planner Agent
planner = Agent(
    role="Content Planner",
    goal=f"Plan engaging and factually accurate content on {topic}",
    backstory=f"""You're working on planning a blog article about the topic: {topic}.
    You collect information that helps the audience learn something and make informed decisions.
    Your work is the basis for the Content Writer to write an article on this topic.""",
    tools=[search_tool],
)

# 2. Content Writer Agent
writer = Agent(
    role="Content Writer",
    goal=f"Write insightful and factually accurate opinion piece about the topic: {topic}",
    backstory=f"""You're working on writing a new opinion piece about the topic: {topic}.
    You base your writing on the work of the Content Planner, who provides an outline and relevant context about the topic.
    You follow the main objectives and direction of the outline, as provided by the Content Planner.
    You also provide objective and impartial insights and back them up with information provide by the Content Planner.
    You acknowledge in your opinion piece when your statements are opinions as opposed to objective statements.""",
)

# 3. Editor Agent
editor = Agent(
    role="Editor",
    goal="Edit a given blog post to align with the writing style of the organization.",
    backstory="""You are an editor who receives a blog post from the Content Writer.
    Your goal is to review the blog post to ensure that it follows journalistic best practices,
    provides balanced viewpoints when providing opinions or assertions, and also avoids major controversial topics
    or opinions when possible.""",
)


# Define tasks for the content creation workflow

# 1. Planning Task
plan_task = Task(
    description=f"""1. Prioritize the latest trends, key players, and noteworthy news on {topic}.
    2. Identify the target audience, considering their interests and pain points.
    3. Develop a detailed content outline including an introduction, key points, and a call to action.
    4. Include SEO keywords and relevant data or sources.""",
    expected_output="A comprehensive content plan document with an outline, audience analysis, and SEO keywords.",
    agent=planner,
)

# 2. Writing Task
write_task = Task(
    description=f"""1. Use the content plan to craft a compelling blog post on {topic}.
    2. Incorporate SEO keywords naturally.
    3. Sections/Subtitles are properly named in an engaging manner.
    4. Ensure the post is structured with an engaging introduction, insightful body, and a summarizing conclusion.
    5. Proofread for grammatical errors and alignment with the brand's voice.""",
    expected_output="A well-written blog post in markdown format, ready for publication, each section should have 2 or 3 paragraphs.",
    agent=writer,
)

# 3. Editing Task
edit_task = Task(
    description="""Proofread the given blog post for grammatical errors and alignment with the brand's voice.""",
    expected_output="A well-written blog post in markdown format, ready for publication, each section should have 2 or 3 paragraphs.",
    agent=editor
)

# Create and execute the content creation crew
content_crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan_task, write_task, edit_task],
    process=Process.sequential,
    verbose=True
)

print("Content creation crew assembled!")
print(f"Agents: {[agent.role for agent in content_crew.agents]}")
print(f"Tasks: {len(content_crew.tasks)}")


result = content_crew.kickoff()

print(result)
