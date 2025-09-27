import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'
os.environ['OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'

llm = LLM(
        model='openai/gpt-4o',
        api_key=os.getenv('OPEN_ROUTER_KEY'),
        base_url="https://openrouter.ai/api/v1"
    )

search_tool = SerperDevTool()
research_agent = Agent(
    role='Senior Research Analyst',

    goal='Uncover cutting-edge developments in AI and data science, providing comprehensive analysis of trends, technologies, and market implications',

    backstory="""You work at a leading tech think tank with over 8 years of experience
    in emerging technology research. Your expertise lies in identifying breakthrough
    innovations before they become mainstream. You have a methodical approach to
    research, always verifying sources and cross-referencing information. You're
    known for your ability to distill complex technical concepts into actionable insights.""",

    llm=llm,
    tools=[search_tool]
)


research_task = Task(
    description="""Conduct a comprehensive analysis of the latest developments in
    Large Language Models (LLMs) and their applications in 2024. Your research should cover:

    1. **Technical Breakthroughs**: Latest architectural innovations, training methodologies,
       and performance improvements in LLMs

    2. **Industry Applications**: Real-world implementations across different sectors
       (healthcare, finance, education, etc.)

    3. **Market Trends**: Investment patterns, major players, and emerging startups
       in the LLM space

    4. **Regulatory Landscape**: Current and upcoming regulations affecting LLM development
       and deployment

    5. **Future Outlook**: Predictions for the next 12-18 months based on current trends

    Use your tools to gather current information from reputable sources, verify claims
    across multiple sources, and provide actionable insights for tech industry stakeholders.""",

    agent=research_agent,

    expected_output="""A comprehensive research report structured as follows:

    # Executive Summary
    - Key findings and main insights (3-4 bullet points)

    # Technical Developments
    - Latest architectural innovations
    - Training methodology improvements
    - Performance benchmarks and comparisons

    # Industry Applications
    - Sector-specific implementations
    - Case studies of successful deployments
    - ROI and impact metrics where available

    # Market Analysis
    - Investment trends and funding patterns
    - Key players and competitive landscape
    - Emerging opportunities and threats

    # Regulatory Environment
    - Current regulations and compliance requirements
    - Upcoming policy changes
    - Geographic variations in regulatory approach

    # Strategic Recommendations
    - Actionable insights for businesses
    - Technology adoption strategies
    - Risk mitigation approaches

    # Future Outlook
    - 12-18 month predictions
    - Emerging trends to watch
    - Potential disruptions

    # Sources and References
    - List of primary sources used
    - Credibility assessment of sources

    Report should be 1500-2000 words, well-structured, and include specific examples
    and data points where possible."""
)




# Create the crew
research_crew = Crew(
    agents=[research_agent],
    tasks=[research_task],
    # custom_llm_provider="openrouter",
    verbose=True,
)

result = research_crew.kickoff()

print(result)