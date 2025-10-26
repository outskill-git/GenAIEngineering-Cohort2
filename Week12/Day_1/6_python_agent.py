from pathlib import Path
from phi.agent.python import PythonAgent
from phi.model.openai import OpenAIChat
from phi.file.local.csv import CsvFile



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



# env_vars_to_clear = ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_API_BASE']
# for var in env_vars_to_clear:
#     if os.getenv(var):
#         print(f"⚠️  Removing conflicting {var}")
#         del os.environ[var]
# os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_KEY")




# Setup working directory
cwd = Path(__file__).parent.resolve()
tmp = cwd.joinpath("tmp")
if not tmp.exists():
    tmp.mkdir(exist_ok=True, parents=True)

# Create Python Agent with IMDB data
eda_agent = PythonAgent(
    name="IMDB EDA Agent",
    model=OpenAIChat(id="gpt-4o"),
    base_dir=tmp,
    files=[
        CsvFile(
            path="https://phidata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv",
            description="Contains comprehensive information about movies from IMDB including ratings, revenue, genres, directors, and actors.",
        )
    ],
    charting_libraries=['plotly', 'matplotlib', 'seaborn', 'pandas'],
    instructions=[
        "You are an expert data analyst specializing in movie industry analysis",
        "Perform comprehensive exploratory data analysis with detailed insights",
        "Create professional, publication-ready visualizations",
        "Always explain your findings and provide actionable insights",
        "Use appropriate statistical methods and visualization techniques",
        "Save all plots and provide clear interpretations"
    ],
    markdown=True,
    pip_install=True,
    show_tool_calls=True,
    save_and_run=True,
)

# Comprehensive EDA request
eda_agent.print_response("""
Please perform a comprehensive exploratory data analysis of the IMDB movie dataset. Include:

**1. Data Overview & Quality Assessment:**
- Dataset shape, column types, and basic info
- Missing values analysis with visualization
- Data quality issues and recommendations
- Statistical summary of all numerical variables

**2. Univariate Analysis:**
- Distribution of movie ratings (histogram, box plot, violin plot)
- Revenue distribution analysis with outlier detection
- Genre frequency analysis with bar charts
- Release year trends over time
- Runtime distribution analysis

**3. Bivariate & Multivariate Analysis:**
- Correlation matrix heatmap for numerical variables
- Rating vs Revenue scatter plot with trend line
- Genre vs Rating box plots
- Director performance analysis (top directors by avg rating)
- Actor performance analysis (if actor data available)
- Year vs Rating trends over decades

**4. Advanced Visualizations:**
- Top 20 highest-rated movies bar chart
- Top 20 highest-grossing movies bar chart
- Genre popularity over time (stacked area chart)
- Rating distribution by decade
- Revenue vs Rating colored by genre (scatter plot)

**5. Statistical Insights:**
- Identify patterns and trends in the data
- Statistical significance tests where appropriate
- Outlier analysis and their characteristics
- Key findings and business insights

**6. Data Export:**
- Save all visualizations as high-quality PNG files
- Create a summary report with key findings
- Export cleaned dataset if any cleaning was performed

Use professional styling for all plots, include proper titles, labels, and legends. Provide detailed interpretations for each visualization.
""", stream=True)