from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew, before_kickoff, after_kickoff
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List, Dict, Any
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['LITELLM_LOG'] = 'DEBUG'
os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'
os.environ['OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'


@CrewBase
class SDLCDevelopmentCrew:
    """
    A comprehensive SDLC crew for building complete product specifications,
    architecture, and development plans for application development.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    # Paths to YAML configuration files
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @before_kickoff
    def prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare and validate inputs before crew execution"""
        # Add default tech stack if not specified
        if 'tech_stack' not in inputs:
            inputs['tech_stack'] = {
                'frontend': 'Streamlit',
                'backend': 'FastAPI',
                'language': 'Python',
                'database': 'PostgreSQL',
                'deployment': 'Docker'
            }

        # Set project timeline
        start_date = datetime.now()
        end_date = start_date + timedelta(days=60)  # 2 months
        inputs['project_timeline'] = {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'duration_weeks': 8
        }

        # Create output directory structure
        inputs['output_directory'] = 'project_deliverables'

        return inputs

    @after_kickoff
    def process_output(self, output):
        """Process and organize output files after crew completion"""
        print("\n" + "="*50)
        print("🎉 SDLC CREW EXECUTION COMPLETED!")
        print("="*50)
        print(f"📁 All deliverables saved to: {output.get('output_directory', 'project_deliverables')}")
        print("\n📋 Generated Documents:")
        print("  ├── business_requirements.md")
        print("  ├── product_requirements.md")
        print("  ├── software_architecture.md")
        print("  ├── api_specifications.yaml")
        print("  ├── high_level_design.md")
        print("  ├── low_level_design.md")
        print("  ├── product_roadmap.md")
        print("  └── tech_stack_analysis.md")
        print("\n✅ Ready for development phase!")

        return output

    # =============================================================================
    # AGENTS DEFINITION
    # =============================================================================

    @agent
    def product_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['product_manager'],
            verbose=True
        )

    @agent
    def business_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['business_analyst'],
            verbose=True
        )

    @agent
    def software_architect(self) -> Agent:
        return Agent(
            config=self.agents_config['software_architect'],
            verbose=True
        )

    @agent
    def technical_lead(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_lead'],
            verbose=True
        )

    @agent
    def api_designer(self) -> Agent:
        return Agent(
            config=self.agents_config['api_designer'],
            verbose=True
        )

    @agent
    def ui_ux_designer(self) -> Agent:
        return Agent(
            config=self.agents_config['ui_ux_designer'],
            verbose=True
        )

    @agent
    def quality_assurance_lead(self) -> Agent:
        return Agent(
            config=self.agents_config['quality_assurance_lead'],
            verbose=True
        )

    # =============================================================================
    # TASKS DEFINITION
    # =============================================================================

    @task
    def analyze_business_requirements(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_business_requirements']
        )

    @task
    def create_product_requirements(self) -> Task:
        return Task(
            config=self.tasks_config['create_product_requirements']
        )

    @task
    def design_software_architecture(self) -> Task:
        return Task(
            config=self.tasks_config['design_software_architecture']
        )

    @task
    def design_api_specifications(self) -> Task:
        return Task(
            config=self.tasks_config['design_api_specifications']
        )

    @task
    def create_high_level_design(self) -> Task:
        return Task(
            config=self.tasks_config['create_high_level_design']
        )

    @task
    def create_low_level_design(self) -> Task:
        return Task(
            config=self.tasks_config['create_low_level_design']
        )

    @task
    def develop_product_roadmap(self) -> Task:
        return Task(
            config=self.tasks_config['develop_product_roadmap']
        )

    @task
    def validate_tech_stack(self) -> Task:
        return Task(
            config=self.tasks_config['validate_tech_stack']
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=False,
            memory=True,
            max_rpm=10,
            share_crew=True
        )


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def run_sdlc_crew(app_idea: str, additional_requirements: str = ""):
    """
    Execute the SDLC crew with the given app idea

    Args:
        app_idea (str): Description of the application to be built
        additional_requirements (str): Any additional specific requirements
    """

    crew_instance = SDLCDevelopmentCrew()

    inputs = {
        'app_idea': app_idea,
        'additional_requirements': additional_requirements,
        'target_users': 'General users',
        'business_goals': 'To be defined during analysis',
        'constraints': 'Budget and time constraints as per timeline'
    }

    result = crew_instance.crew().kickoff(inputs=inputs)
    return result

# Example usage:
if __name__ == "__main__":
    app_description = """
    A task management application that helps teams collaborate on projects.
    Key features should include:
    - User authentication and role management
    - Project creation and management
    - Task assignment and tracking
    - Real-time notifications
    - Dashboard with analytics
    - Mobile-responsive design
    """

    additional_reqs = """
    - Must integrate with popular calendar applications
    - Should support file attachments
    - Needs offline capability for mobile users
    - Must comply with GDPR requirements
    """

    result = run_sdlc_crew(app_description, additional_reqs)