import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from crewai.tools import tool

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'
os.environ['OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'

# Creating a specialized agent team
z
# 1. Research Specialist
researcher = Agent(
    role='Senior Research Specialist',
    goal='Conduct thorough research on any topic and compile comprehensive findings',
    backstory="""You're a meticulous researcher with expertise in multiple domains.
    You excel at finding reliable sources, fact-checking information, and organizing
    research findings in a structured manner.""",
    tools=[search_tool],
)

# 2. Data Analyst
analyst = Agent(
    role='Lead Data Analyst',
    goal='Analyze data patterns and extract meaningful insights for decision-making',
    backstory="""You're an experienced data analyst with strong statistical background.
    You can work with various data formats and always validate your analytical
    approaches before drawing conclusions.""",
)

# 3. Content Creator
writer = Agent(
    role='Expert Content Creator',
    goal='Transform complex information into engaging, accessible content',
    backstory="""You're a skilled writer with the ability to adapt your style to
    different audiences. You excel at structuring information logically and
    making complex topics understandable.""",
)

# 4. Quality Reviewer
reviewer = Agent(
    role='Senior Quality Reviewer',
    goal='Ensure all outputs meet high standards of accuracy, clarity, and completeness',
    backstory="""You're a detail-oriented professional with years of experience in
    quality assurance. You have a keen eye for errors and always provide
    constructive feedback for improvements.""",
)

# Store our agent team
agent_team = {
    'researcher': researcher,
    'analyst': analyst,
    'writer': writer,
    'reviewer': reviewer
}


research_task = Task(
    description="Research the latest trends in AI-powered chatbots for 2024",
    agent=researcher,
    expected_output="A comprehensive research report on chatbot trends",

)

writer_task = Task(
    description="write on research topic",
    agent=writer,
    expected_output="well researched and structured content on the topic",
    context=[research_task]  # Reference the actual task object
)

review_task = Task(
    description="review the content for accuracy and clarity",
    agent=reviewer,
    expected_output="reviewed content with feedback for improvements",
    context=[research_task, writer_task]  # Reference both task objects
)

llm = LLM(
        model='openai/gpt-4o',
        api_key=os.getenv('OPEN_ROUTER_KEY'),
        base_url="https://openrouter.ai/api/v1"
    )

# Create the crew
research_crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writer_task, review_task],
    llm=llm,
    custom_llm_provider="openrouter",
    verbose=True,
)


result = research_crew.kickoff()

print(result)