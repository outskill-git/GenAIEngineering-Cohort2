
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
# Create a research agent
research_agent = Agent(
    role='Senior Market Research Analyst',
    goal=f'Conduct thorough market research and provide actionable business insights for startup ventures in {startup_idea}',
    backstory="""You are an experienced market researcher with 10+ years experience.
    You excel at identifying market trends, competitive landscapes,
    and growth opportunities. Your reports are known for their depth and actionable recommendations.""",
    tools=[search_tool, file_tool],
    verbose=True,
    memory=True
)


review_agent = Agent(
    role='Senior Quality Reviewer',
    goal='Review and provide feedback for continuous improvement',
    backstory="Expert reviewer with high standards for quality and accuracy.",
    tools=[file_tool],
    verbose=True
)


analysis_task=Task(
        description="Create initial market analysis draft",
        agent=research_agent,
        expected_output="Initial analysis draft",
    )

review_task=Task(
        description="Review the analysis and provide detailed feedback",
        agent=review_agent,
        expected_output="Detailed review with improvement suggestions",
        context=[analysis_task]
    )

refine_task=Task(
        description="Refine the analysis based on reviewer feedback",
        agent=research_agent,
        expected_output="Refined analysis incorporating feedback",
        context=[analysis_task, review_task]
    )

quality_task=Task(
        description="Final quality check and approval",
        agent=review_agent,
        expected_output="Final approved analysis",
        context=[refine_task]
    )


content_crew = Crew(
    agents=[research_agent, review_agent],
    tasks=[analysis_task, review_task, refine_task, quality_task],
    verbose=True
)

result = content_crew.kickoff()
print(result)

