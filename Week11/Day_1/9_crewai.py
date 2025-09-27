import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM, Process
from crewai_tools import FileReadTool

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'
os.environ['OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'

llm = LLM(
        model='openai/gpt-4o',
        api_key=os.getenv('OPEN_ROUTER_KEY'),
        base_url="https://openrouter.ai/api/v1"
    )


# Check if text file exists
txt_file = 'IndianBudget2025.txt'

if not os.path.exists(txt_file):
    print(f"⚠️  Warning: Text file '{txt_file}' not found in current directory")
    print("Please convert your PDF to text first or create the text file manually")

    # Try to convert PDF if it exists
    pdf_file = 'IndianBudget2025.pdf'
    if os.path.exists(pdf_file):
        print(f"Found PDF file. Attempting to convert...")
        try:
            import PyPDF2

            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"

            with open(txt_file, 'w', encoding='utf-8') as txt_file_write:
                txt_file_write.write(text)

            print(f"✅ Successfully converted {pdf_file} to {txt_file}")

        except ImportError:
            print("PyPDF2 not installed. Install it with: pip install PyPDF2")
            exit()
        except Exception as e:
            print(f"Error converting PDF: {e}")
            exit()
    else:
        print("No PDF file found either. Please provide the budget document.")
        exit()
else:
    print(f"✅ Text file '{txt_file}' found")

# Initialize File Reading Tool
file_tool = FileReadTool()

# Create research analyst agent
research_analyst = Agent(
    role='Senior Macro Economist and Budget Analyst',
    goal='Analyze the Indian Budget 2025 document and provide comprehensive economic insights and strategic recommendations',
    backstory="""You are a highly experienced macro economist with over 20 years of expertise
    in analyzing government budgets, fiscal policies, and their macroeconomic implications.
    You have specialized knowledge of Indian economic policy, taxation systems, and
    public finance. You excel at breaking down complex budget documents into actionable
    insights for policymakers, investors, and businesses.""",
    tools=[file_tool],
    llm=llm,
    verbose=True
)

# Create comprehensive budget analysis task
budget_analysis_task = Task(
    description=f"""Read and analyze the Indian Budget 2025 document from {txt_file}.
    Perform a comprehensive economic analysis covering:

    **FISCAL ANALYSIS:**
    1. Total budget size and key fiscal targets
    2. Revenue projections (tax and non-tax revenue)
    3. Expenditure breakdown (capital vs revenue expenditure)
    4. Fiscal deficit, revenue deficit, and primary deficit targets
    5. Debt-to-GDP ratio and debt management strategy

    **SECTORAL ALLOCATION:**
    1. Infrastructure development allocations
    2. Healthcare and medical research funding
    3. Education and skill development investments
    4. Defense and security expenditure
    5. Agriculture and rural development schemes
    6. Social welfare and subsidy allocations

    **TAX POLICY CHANGES:**
    1. Income tax slab modifications
    2. Corporate tax rate changes
    3. GST rate adjustments
    4. Customs and excise duty changes
    5. New tax incentives or exemptions

    **ECONOMIC IMPACT:**
    1. Expected GDP growth impact
    2. Inflation projections and management
    3. Employment generation potential
    4. Impact on different income groups
    5. Regional development implications

    **STRATEGIC RECOMMENDATIONS:**
    1. Investment opportunities for businesses
    2. Policy implementation challenges
    3. Market implications for different sectors
    4. Long-term economic growth prospects

    Use the file reading tool to extract specific data, figures, and policy details from the budget document.""",

    expected_output="""A comprehensive budget analysis report structured as follows:

    # Indian Budget 2025: Economic Analysis Report

    ## Executive Summary
    - Key budget highlights (3-5 major points)
    - Overall fiscal stance and economic philosophy
    - Major policy shifts from previous year

    ## Fiscal Overview
    - Budget size and growth from previous year
    - Revenue and expenditure breakdown with percentages
    - Deficit targets and debt management strategy
    - Specific fiscal numbers and ratios

    ## Sectoral Analysis
    - Detailed allocation breakdown by sector
    - Year-over-year changes in major sectors
    - New schemes and policy initiatives
    - Infrastructure and development priorities

    ## Tax Policy Analysis
    - All tax rate changes with specific details
    - Impact on individuals and businesses
    - New exemptions, deductions, or incentives
    - Revenue implications of tax changes

    ## Economic Impact Assessment
    - Growth projections and assumptions
    - Inflation management measures
    - Employment and development impact
    - Regional and demographic implications

    ## Investment and Business Implications
    - Sectoral opportunities for investors
    - Policy changes affecting business operations
    - Market outlook for different industries
    - Strategic recommendations for stakeholders

    ## Conclusion and Outlook
    - Overall assessment of budget effectiveness
    - Key risks and implementation challenges
    - Long-term economic development trajectory

    The report should be data-driven with specific numbers, percentages, and allocations cited from the budget document. Include comparative analysis where previous year data is available.""",

    agent=research_analyst,
    output_file="indian_budget_2025_comprehensive_analysis.md"
)

# Create and execute the crew
budget_crew = Crew(
    agents=[research_analyst],
    tasks=[budget_analysis_task],
    process=Process.sequential,
    verbose=True
)

# Execute the analysis
print(f"\n🚀 Starting comprehensive analysis of Indian Budget 2025...")
print("=" * 60)

try:
    result = budget_crew.kickoff()

    print("\n✅ Budget analysis completed successfully!")
    print("=" * 60)
    print("\n📊 Analysis Results:")
    print(result)
    print("=" * 60)
    print(f"\n💾 Detailed analysis saved to: indian_budget_2025_comprehensive_analysis.md")

    # Additional summary
    print(f"\n📄 Source document: {txt_file}")
    print("📈 Analysis includes fiscal overview, sectoral breakdown, tax changes, and economic impact")

except Exception as e:
    print(f"\n❌ Error during analysis: {str(e)}")
    print("Please check your text file and API configuration")
    print("Troubleshooting tips:")
    print("1. Ensure the text file exists and is readable")
    print("2. Check your OpenRouter API key configuration")
    print("3. Verify the LLM model name is correct")