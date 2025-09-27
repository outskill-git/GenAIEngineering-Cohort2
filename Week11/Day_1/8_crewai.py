import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM, Process
from crewai_tools import SerperDevTool, FileReadTool

load_dotenv()


# Environment variables (backup configuration)
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'
os.environ['OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'

llm = LLM(
        model='openai/gpt-4o',
        api_key=os.getenv('OPEN_ROUTER_KEY'),
        base_url="https://openrouter.ai/api/v1"
    )

search_tool = SerperDevTool()
regional_analysts = {
    'north_america': Agent(
        role='North America Market Analyst',
        goal='Analyze AI customer service market trends in North America',
        backstory="Expert in North American technology markets with deep understanding of regulatory environment and customer preferences.",
        tools=[search_tool],
        llm=llm,
        verbose=True
    ),
    'europe': Agent(
        role='European Market Analyst',
        goal='Analyze AI customer service market trends in Europe',
        backstory="Specialist in European markets with expertise in GDPR compliance and diverse cultural market dynamics.",
        tools=[search_tool],
        llm=llm,
        verbose=True
    ),
    'asia_pacific': Agent(
        role='Asia-Pacific Market Analyst',
        goal='Analyze AI customer service market trends in Asia-Pacific',
        backstory="Expert in APAC markets with understanding of rapid technology adoption and diverse regulatory landscapes.",
        tools=[search_tool],
        llm=llm,
        verbose=True
    )
}

parallel_tasks = []
for region, agent in regional_analysts.items():
    task = Task(
        description=f"""Conduct regional market analysis for {region.replace('_', ' ').title()}
        focusing on AI customer service automation. Include:

        1. Regional market size and growth rates
        2. Key local and international players
        3. Regulatory environment and compliance requirements
        4. Cultural factors affecting adoption
        5. Technology infrastructure readiness
        6. Customer preferences and behavior patterns
        7. Regional opportunities and challenges

        Provide region-specific insights and recommendations.""",

        agent=agent,

        expected_output=f"""Regional analysis report for {region.replace('_', ' ').title()} including:
        - Market overview and sizing
        - Competitive landscape
        - Regulatory considerations
        - Cultural and adoption factors
        - Strategic recommendations for market entry

        Report should be 1000-1500 words with region-specific insights.""",

        async_execution = True,
        output_file=f"{region}_market_analysis.md",
        verbose=True
    )

    parallel_tasks.append(task)

strategy_consultant = Agent(
    role='market strategy consultant',
    goal='Develop comprehensive business strategies based on market analysis and data insights',
    backstory="""You are a senior strategy consultant with expertise in technology
    market entry strategies. You excel at synthesizing research and data into
    actionable business plans and strategic recommendations.""",
    llm=llm,
    verbose=True
)

synthesis_task = Task(
    description="Synthesize all research findings",
    agent=strategy_consultant,
    expected_output="Integrated strategic analysis and recommendations",
    context=parallel_tasks
)

content_crew = Crew(
    agents=list(regional_analysts.values()) + [strategy_consultant],
    tasks=parallel_tasks + [synthesis_task],  # Changed order: parallel tasks first, then synthesis
    llm=llm,
    verbose=True
)

result = content_crew.kickoff()
print(result)