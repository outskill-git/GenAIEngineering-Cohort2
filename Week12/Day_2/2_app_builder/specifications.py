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

# Clear conflicting environment variables
env_vars_to_clear = ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_API_BASE']
for var in env_vars_to_clear:
    if os.getenv(var):
        print(f"⚠️  Removing conflicting {var}")
        del os.environ[var]
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_KEY")

import json
import yaml
from pathlib import Path
from typing import Dict, Any

from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.file import FileTools


class SpecificationGenerator:
    """
    Uses Phidata agents to generate specification files based on product requirements
    """

    def __init__(self, output_dir: str = "specifications"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Product Analysis Agent
        self.product_analyst = Agent(
            model=OpenAIChat(id="gpt-4o"),
            name="Product Analyst",
            description="Expert in analyzing product requirements and creating detailed specifications",
            instructions=[
                "Analyze product requirements thoroughly",
                "Extract key features, user needs, and technical requirements",
                "Identify the main entities and relationships in the product",
                "Consider scalability, security, and user experience",
                "Create comprehensive and realistic specifications"
            ],
            tools=[FileTools()],
            markdown=True,
            show_tool_calls=True
        )

        # API Specification Agent
        self.api_designer = Agent(
            model=OpenAIChat(id="gpt-4o"),
            name="API Designer",
            description="Expert in designing RESTful APIs and OpenAPI specifications",
            instructions=[
                "Design comprehensive REST APIs based on product requirements",
                "Create detailed OpenAPI 3.0 specifications",
                "Include proper HTTP methods, status codes, and error handling",
                "Design appropriate request/response schemas",
                "Consider API security, validation, and best practices",
                "Ensure APIs support all required product functionality"
            ],
            tools=[FileTools()],
            markdown=True,
            show_tool_calls=True
        )

        # Database Designer Agent
        self.db_designer = Agent(
            model=OpenAIChat(id="gpt-4o"),
            name="Database Designer",
            description="Expert in designing database schemas and data models",
            instructions=[
                "Design normalized database schemas based on product requirements",
                "Create appropriate tables, relationships, and constraints",
                "Consider data integrity, performance, and scalability",
                "Include proper indexing strategies",
                "Design for both OLTP and potential analytics needs",
                "Follow database design best practices"
            ],
            tools=[FileTools()],
            markdown=True,
            show_tool_calls=True
        )

        # UI/UX Designer Agent
        self.ui_designer = Agent(
            model=OpenAIChat(id="gpt-4o"),
            name="UI/UX Designer",
            description="Expert in designing user interfaces and user experiences",
            instructions=[
                "Design intuitive and user-friendly interfaces",
                "Create comprehensive UI component specifications",
                "Consider accessibility, responsiveness, and usability",
                "Design user flows and interaction patterns",
                "Specify modern design systems and visual hierarchy",
                "Focus on user experience and conversion optimization"
            ],
            tools=[FileTools()],
            markdown=True,
            show_tool_calls=True
        )

    def generate_specifications(self, product_requirement: str, product_name: str = "MyProduct"):
        """
        Generate all specification files using AI agents

        Args:
            product_requirement: Description of what the product should do
            product_name: Name of the product
        """
        print(f"🤖 Starting AI-powered specification generation...")
        print(f"📝 Product: {product_name}")
        print(f"📋 Requirement: {product_requirement}")
        print(f"📁 Output directory: {self.output_dir}")

        # Step 1: Generate product idea and analysis
        print("\n📊 Analyzing product requirements...")
        self.generate_product_idea(product_requirement, product_name)

        # Step 2: Generate API specification
        print("\n🔌 Designing API specification...")
        self.generate_api_spec(product_requirement, product_name)

        # Step 3: Generate database specification
        print("\n🗄️ Designing database schema...")
        self.generate_database_spec(product_requirement, product_name)

        # Step 4: Generate UI specification
        print("\n🎨 Designing UI specification...")
        self.generate_ui_spec(product_requirement, product_name)

        # Step 5: Generate additional specifications
        print("\n⚙️ Creating additional specifications...")
        self.generate_agent_spec(product_requirement, product_name)

        print(f"\n✅ All AI-generated specification files created in {self.output_dir}/")
        print("📋 Files created:")
        for file in self.output_dir.glob("*"):
            print(f"  - {file.name}")

    def generate_product_idea(self, requirement: str, name: str):
        """Use AI agent to generate product_idea.json"""
        prompt = f"""
        Analyze this product requirement and create a comprehensive product specification:

        Product Name: {name}
        Requirement: {requirement}

        Create a detailed JSON specification that includes:
        1. Product name and description
        2. Key features (extract and expand from the requirement)
        3. Technical requirements (choose appropriate tech stack)
        4. Target audience analysis
        5. Success metrics
        6. User personas
        7. Business objectives
        8. Constraints and assumptions

        Make the specification realistic, comprehensive, and actionable.
        Format as valid JSON.
        """

        response = self.product_analyst.run(prompt)

        # Extract JSON from response and save
        try:
            # Try to find JSON in the response
            content = response.content
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            else:
                # Look for JSON object
                start = content.find("{")
                end = content.rfind("}") + 1
                json_content = content[start:end]

            # Parse and save
            product_spec = json.loads(json_content)

            with open(self.output_dir / "product_idea.json", 'w') as f:
                json.dump(product_spec, f, indent=2)
            print("✅ Created product_idea.json")

        except Exception as e:
            print(f"❌ Error generating product_idea.json: {e}")
            # Create fallback specification
            fallback_spec = {
                "name": name,
                "description": requirement,
                "key_features": ["Core Functionality", "User Management", "Data Processing"],
                "technical_requirements": {
                    "backend": "FastAPI",
                    "frontend": "React",
                    "database": "PostgreSQL"
                }
            }
            with open(self.output_dir / "product_idea.json", 'w') as f:
                json.dump(fallback_spec, f, indent=2)
            print("✅ Created fallback product_idea.json")

    def generate_api_spec(self, requirement: str, name: str):
        """Use AI agent to generate openapi.yaml"""
        prompt = f"""
        Design a comprehensive REST API for this product:

        Product Name: {name}
        Requirement: {requirement}

        Create a complete OpenAPI 3.0 specification that includes:
        1. API info (title, description, version)
        2. All necessary endpoints for the product functionality
        3. Proper HTTP methods (GET, POST, PUT, DELETE)
        4. Request/response schemas
        5. Authentication scheme (JWT)
        6. Error responses
        7. Query parameters and path parameters
        8. Comprehensive data models in components/schemas

        Make sure the API covers all aspects mentioned in the requirement.
        Output as valid OpenAPI YAML format.
        """

        response = self.api_designer.run(prompt)

        try:
            # Extract YAML from response
            content = response.content
            if "```yaml" in content:
                yaml_start = content.find("```yaml") + 7
                yaml_end = content.find("```", yaml_start)
                yaml_content = content[yaml_start:yaml_end].strip()
            elif "```" in content:
                yaml_start = content.find("```") + 3
                yaml_end = content.find("```", yaml_start)
                yaml_content = content[yaml_start:yaml_end].strip()
            else:
                yaml_content = content

            # Parse and save YAML
            api_spec = yaml.safe_load(yaml_content)

            with open(self.output_dir / "openapi.yaml", 'w') as f:
                yaml.dump(api_spec, f, default_flow_style=False, sort_keys=False)
            print("✅ Created openapi.yaml")

        except Exception as e:
            print(f"❌ Error generating openapi.yaml: {e}")
            # Create minimal fallback
            fallback_api = {
                "openapi": "3.0.3",
                "info": {"title": f"{name} API", "version": "1.0.0", "description": requirement},
                "paths": {
                    "/api/v1/items": {
                        "get": {"summary": "Get items", "responses": {"200": {"description": "Success"}}}
                    }
                }
            }
            with open(self.output_dir / "openapi.yaml", 'w') as f:
                yaml.dump(fallback_api, f, default_flow_style=False)
            print("✅ Created fallback openapi.yaml")

    def generate_database_spec(self, requirement: str, name: str):
        """Use AI agent to generate database_specification.json"""
        prompt = f"""
        Design a comprehensive database schema for this product:

        Product Name: {name}
        Requirement: {requirement}

        Create a detailed database specification that includes:
        1. Database type (PostgreSQL recommended)
        2. All necessary tables for the product functionality
        3. Columns with appropriate data types
        4. Primary keys, foreign keys, and constraints
        5. Indexes for performance optimization
        6. Relationships between tables
        7. Sample data requirements
        8. Consider scalability and normalization

        Analyze the requirement thoroughly to identify all entities and relationships.
        Format as valid JSON.
        """

        response = self.db_designer.run(prompt)

        try:
            # Extract JSON from response
            content = response.content
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            else:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_content = content[start:end]

            db_spec = json.loads(json_content)

            with open(self.output_dir / "database_specification.json", 'w') as f:
                json.dump(db_spec, f, indent=2)
            print("✅ Created database_specification.json")

        except Exception as e:
            print(f"❌ Error generating database_specification.json: {e}")
            # Create fallback
            fallback_db = {
                "database_type": "postgresql",
                "tables": [
                    {
                        "name": "users",
                        "columns": [
                            {"name": "id", "type": "SERIAL PRIMARY KEY"},
                            {"name": "username", "type": "VARCHAR(50) UNIQUE NOT NULL"},
                            {"name": "email", "type": "VARCHAR(100) UNIQUE NOT NULL"}
                        ]
                    }
                ]
            }
            with open(self.output_dir / "database_specification.json", 'w') as f:
                json.dump(fallback_db, f, indent=2)
            print("✅ Created fallback database_specification.json")

    def generate_ui_spec(self, requirement: str, name: str):
        """Use AI agent to generate ui_specification.json"""
        prompt = f"""
        Design a comprehensive user interface specification for this product:

        Product Name: {name}
        Requirement: {requirement}

        Create a detailed UI/UX specification that includes:
        1. Design system and visual guidelines
        2. Component library (headers, forms, buttons, etc.)
        3. Page layouts and user flows
        4. Responsive design considerations
        5. Accessibility requirements
        6. Color scheme and typography
        7. User interaction patterns
        8. Dashboard and admin interfaces if needed

        Consider modern web design principles and user experience best practices.
        Format as valid JSON.
        """

        response = self.ui_designer.run(prompt)

        try:
            # Extract JSON from response
            content = response.content
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            else:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_content = content[start:end]

            ui_spec = json.loads(json_content)

            with open(self.output_dir / "ui_specification.json", 'w') as f:
                json.dump(ui_spec, f, indent=2)
            print("✅ Created ui_specification.json")

        except Exception as e:
            print(f"❌ Error generating ui_specification.json: {e}")
            # Create fallback
            fallback_ui = {
                "design_system": "modern",
                "components": [
                    {"name": "Header", "type": "navigation"},
                    {"name": "Dashboard", "type": "page"}
                ],
                "theme": {"colors": "blue", "typography": "sans-serif"}
            }
            with open(self.output_dir / "ui_specification.json", 'w') as f:
                json.dump(fallback_ui, f, indent=2)
            print("✅ Created fallback ui_specification.json")

    def generate_agent_spec(self, requirement: str, name: str):
        """Generate additional specifications using AI analysis"""

        # Generate API specification (simpler format)
        api_spec = {
            "base_url": "http://localhost:8000/api/v1",
            "authentication": "JWT Bearer Token",
            "endpoints": [
                {"path": "/items", "methods": ["GET", "POST"], "description": "Manage items"},
                {"path": "/items/{id}", "methods": ["GET", "PUT", "DELETE"], "description": "Individual item operations"}
            ]
        }

        with open(self.output_dir / "api_specification.json", 'w') as f:
            json.dump(api_spec, f, indent=2)
        print("✅ Created api_specification.json")

        # Generate agent specification for AI features
        agent_spec = {
            "agents": [
                {
                    "name": "DataProcessingAgent",
                    "description": "Processes and validates data based on product requirements",
                    "capabilities": ["data_validation", "processing", "analysis"]
                }
            ],
            "workflows": [
                {
                    "name": "MainWorkflow",
                    "description": "Primary workflow for the product",
                    "steps": ["input", "process", "output"]
                }
            ]
        }

        with open(self.output_dir / "agent_specification.json", 'w') as f:
            json.dump(agent_spec, f, indent=2)
        print("✅ Created agent_specification.json")


# Main function to generate specifications
def generate_specifications_with_ai(
    product_requirement: str,
    product_name: str = "MyProduct",
    output_dir: str = "specifications"
):
    """
    Generate comprehensive product specifications using AI agents

    Args:
        product_requirement: Description of what the product should do
        product_name: Name of the product
        output_dir: Directory to save specification files
    """

    print("🤖 Starting AI-powered specification generation...")

    generator = SpecificationGenerator(output_dir)
    generator.generate_specifications(product_requirement, product_name)

    print(f"\n🎉 Specification generation completed!")
    print(f"📁 All files saved in: {output_dir}/")
    print(f"\n📋 Generated specifications:")
    print(f"  - product_idea.json       # Product analysis and requirements")
    print(f"  - openapi.yaml           # Complete API specification")
    print(f"  - database_specification.json  # Database schema design")
    print(f"  - ui_specification.json  # UI/UX design specifications")
    print(f"  - api_specification.json # API summary")
    print(f"  - agent_specification.json # AI agent configurations")


# Example usage
if __name__ == "__main__":
    # Example product requirement
    example_requirement = """
    Create a task management system that allows users to create, edit, and track tasks.
    Users should be able to set due dates, priorities, and assign tasks to team members.
    The system should have a dashboard showing task progress, notifications for deadlines,
    and the ability to generate reports on productivity.
    """

    generate_specifications_with_ai(
        product_requirement=example_requirement,
        product_name="TaskMaster Pro",
        output_dir="specifications"
    )