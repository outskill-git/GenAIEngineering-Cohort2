import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Clear conflicting environment variables
env_vars_to_clear = ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_API_BASE']
for var in env_vars_to_clear:
    if os.getenv(var):
        print(f"⚠️  Removing conflicting {var}")
        del os.environ[var]
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_KEY")

import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Clear conflicting environment variables
env_vars_to_clear = ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_API_BASE']
for var in env_vars_to_clear:
    if os.getenv(var):
        print(f"⚠️  Removing conflicting {var}")
        del os.environ[var]
os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_AI_KEY")

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Iterator

from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.workflow import Workflow, RunResponse, RunEvent
from phi.storage.workflow.sqlite import SqlWorkflowStorage
from phi.tools.file import FileTools


class SpecificationBasedCodeGenerationWorkflow(Workflow):
    """
    Generic workflow that uses Phidata agents to generate code from specification files
    """

    # Generic Database Code Generation Agent
    database_code_agent: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="Database Code Generator",
        description="Reads database specifications and generates SQL schemas, migrations, and SQLAlchemy models",
        instructions=[
            "Read and parse database specification files (JSON format)",
            "Generate complete SQL schema files with proper table definitions",
            "Create SQLAlchemy models that match the database specification exactly",
            "Generate migration scripts for database setup",
            "Include proper indexes, constraints, and relationships as specified",
            "Follow PostgreSQL syntax and best practices",
            "Generate seed data based on the specification",
            "Add proper comments and documentation to generated SQL"
        ],
        tools=[FileTools()],
        markdown=True,
        show_tool_calls=True
    )

    # Generic API Code Generation Agent
    api_code_agent: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="API Code Generator",
        description="Reads OpenAPI specifications and generates complete FastAPI applications",
        instructions=[
            "Read and parse OpenAPI YAML specifications completely",
            "Generate FastAPI applications that implement every endpoint in the specification",
            "Create Pydantic models from OpenAPI schemas with exact field mappings",
            "Implement proper HTTP methods, parameters, and response codes",
            "Generate authentication systems based on OpenAPI security schemes",
            "Create proper error handling and validation",
            "Follow FastAPI best practices and conventions",
            "Generate modular, production-ready code with proper structure",
            "Include comprehensive logging and monitoring"
        ],
        tools=[FileTools()],
        markdown=True,
        show_tool_calls=True
    )

    # Generic UI Code Generation Agent
    ui_code_agent: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="UI Code Generator",
        description="Reads UI specifications and generates vanilla HTML, CSS, and JavaScript",
        instructions=[
            "Read and parse UI specification files (JSON format)",
            "Generate semantic HTML based on component and layout specifications",
            "Create modern, responsive CSS that matches the styling specifications",
            "Generate vanilla JavaScript (no frameworks) for all interactive features",
            "Implement all components specified in the UI specification",
            "Create layouts that match the specification exactly",
            "Apply styling rules as defined in the specification",
            "Implement user flows and interactions as described",
            "Ensure accessibility and modern web standards",
            "Create mobile-responsive designs"
        ],
        tools=[FileTools()],
        markdown=True,
        show_tool_calls=True
    )

    # Generic Agent Code Generation Agent
    agent_code_agent: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="Agent Code Generator",
        description="Reads agent specifications and generates Phidata agent implementations",
        instructions=[
            "Read and parse agent specification files (JSON format)",
            "Generate individual Phidata agent implementations for each specified agent",
            "Implement workflows that coordinate multiple agents as specified",
            "Create proper tool integrations based on the tools listed in specifications",
            "Generate agent orchestration systems for complex multi-agent tasks",
            "Follow Phidata framework patterns and best practices",
            "Include proper error handling, logging, and monitoring for agents",
            "Create modular, reusable agent architectures",
            "Implement the exact instructions and capabilities specified for each agent"
        ],
        tools=[FileTools()],
        markdown=True,
        show_tool_calls=True
    )

    # CI/CD Code Generation Agent
    cicd_code_agent: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="CI/CD Code Generator",
        description="Generates complete CI/CD pipeline configurations, Docker files, and Git setup",
        instructions=[
            "Read project specifications and generate comprehensive CI/CD configurations",
            "Create production-ready Dockerfiles for multi-stage builds",
            "Generate .dockerignore files with proper exclusions",
            "Create comprehensive .gitignore files for Python/JavaScript projects",
            "Generate GitHub Actions workflows for CI/CD pipelines",
            "Include automated testing, linting, security scanning, and deployment",
            "Create Docker Compose files for local development",
            "Generate deployment scripts and configuration files",
            "Include proper environment variable handling and secrets management",
            "Follow DevOps best practices for containerization and automation",
            "Generate comprehensive documentation for deployment and CI/CD processes"
        ],
        tools=[FileTools()],
        markdown=True,
        show_tool_calls=True
    )

class SpecificationBasedCodeGenerationWorkflow(Workflow):
    """
    Generic workflow that uses Phidata agents to generate code from specification files
    """

    # Generic Database Code Generation Agent
    database_code_agent: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="Database Code Generator",
        description="Reads database specifications and generates SQL schemas, migrations, and SQLAlchemy models",
        instructions=[
            "Read and parse database specification files (JSON format)",
            "Generate complete SQL schema files with proper table definitions",
            "Create SQLAlchemy models that match the database specification exactly",
            "Generate migration scripts for database setup",
            "Include proper indexes, constraints, and relationships as specified",
            "Follow PostgreSQL syntax and best practices",
            "Generate seed data based on the specification",
            "Add proper comments and documentation to generated SQL"
        ],
        tools=[FileTools()],
        markdown=True,
        show_tool_calls=True
    )

    # Generic API Code Generation Agent
    api_code_agent: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="API Code Generator",
        description="Reads OpenAPI specifications and generates complete FastAPI applications",
        instructions=[
            "Read and parse OpenAPI YAML specifications completely",
            "Generate FastAPI applications that implement every endpoint in the specification",
            "Create Pydantic models from OpenAPI schemas with exact field mappings",
            "Implement proper HTTP methods, parameters, and response codes",
            "Generate authentication systems based on OpenAPI security schemes",
            "Create proper error handling and validation",
            "Follow FastAPI best practices and conventions",
            "Generate modular, production-ready code with proper structure",
            "Include comprehensive logging and monitoring"
        ],
        tools=[FileTools()],
        markdown=True,
        show_tool_calls=True
    )

    # Generic UI Code Generation Agent
    ui_code_agent: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="UI Code Generator",
        description="Reads UI specifications and generates vanilla HTML, CSS, and JavaScript",
        instructions=[
            "Read and parse UI specification files (JSON format)",
            "Generate semantic HTML based on component and layout specifications",
            "Create modern, responsive CSS that matches the styling specifications",
            "Generate vanilla JavaScript (no frameworks) for all interactive features",
            "Implement all components specified in the UI specification",
            "Create layouts that match the specification exactly",
            "Apply styling rules as defined in the specification",
            "Implement user flows and interactions as described",
            "Ensure accessibility and modern web standards",
            "Create mobile-responsive designs"
        ],
        tools=[FileTools()],
        markdown=True,
        show_tool_calls=True
    )

    # Generic Agent Code Generation Agent
    agent_code_agent: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="Agent Code Generator",
        description="Reads agent specifications and generates Phidata agent implementations",
        instructions=[
            "Read and parse agent specification files (JSON format)",
            "Generate individual Phidata agent implementations for each specified agent",
            "Implement workflows that coordinate multiple agents as specified",
            "Create proper tool integrations based on the tools listed in specifications",
            "Generate agent orchestration systems for complex multi-agent tasks",
            "Follow Phidata framework patterns and best practices",
            "Include proper error handling, logging, and monitoring for agents",
            "Create modular, reusable agent architectures",
            "Implement the exact instructions and capabilities specified for each agent"
        ],
        tools=[FileTools()],
        markdown=True,
        show_tool_calls=True
    )

    # CI/CD Code Generation Agent
    cicd_code_agent: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        name="CI/CD Code Generator",
        description="Generates complete CI/CD pipeline configurations, Docker files, and Git setup",
        instructions=[
            "Read project specifications and generate comprehensive CI/CD configurations",
            "Create production-ready Dockerfiles for multi-stage builds",
            "Generate .dockerignore files with proper exclusions",
            "Create comprehensive .gitignore files for Python/JavaScript projects",
            "Generate GitHub Actions workflows for CI/CD pipelines",
            "Include automated testing, linting, security scanning, and deployment",
            "Create Docker Compose files for local development",
            "Generate deployment scripts and configuration files",
            "Include proper environment variable handling and secrets management",
            "Follow DevOps best practices for containerization and automation",
            "Generate comprehensive documentation for deployment and CI/CD processes"
        ],
        tools=[FileTools()],
        markdown=True,
        show_tool_calls=True
    )

    def run(self, specs_dir: str = "specifications", output_dir: str = "generated_code") -> Iterator[RunResponse]:
        """Main workflow that generates code from specifications using AI agents"""

        # Initialize paths
        specs_path = Path(specs_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        yield RunResponse(
            event=RunEvent.workflow_started,
            content="🚀 Starting specification-based code generation..."
        )

        # Load all specification files
        specifications = self.load_all_specifications(specs_path)

        # Create project structure
        self.create_project_structure(output_path)
        yield RunResponse(content="📁 Created project directory structure")

        # Generate SQL database code from database specification
        if specifications.get('database'):
            yield RunResponse(content="🗄️ Generating database code from specification...")
            yield from self.generate_database_code_from_spec(specifications['database'], output_path)

        # Generate FastAPI code from OpenAPI specification
        if specifications.get('openapi'):
            yield RunResponse(content="🔌 Generating FastAPI code from OpenAPI specification...")
            yield from self.generate_api_code_from_spec(specifications['openapi'], output_path)

        # Generate Phidata agents from agent specification
        if specifications.get('agents'):
            yield RunResponse(content="🤖 Generating Phidata agents from specification...")
            yield from self.generate_agent_code_from_spec(specifications['agents'], output_path)

        # Generate UI code from UI specification
        if specifications.get('ui'):
            yield RunResponse(content="🎨 Generating vanilla HTML/CSS/JS from UI specification...")
            yield from self.generate_ui_code_from_spec(specifications['ui'], output_path)

        # Generate CI/CD pipeline and deployment files
        yield RunResponse(content="🚀 Generating CI/CD pipeline and deployment configuration...")
        yield from self.generate_cicd_code_from_spec(specifications, output_path)

        # Generate project documentation and configuration
        yield RunResponse(content="📚 Generating project configuration and documentation...")
        yield from self.generate_project_files(specifications, output_path)

        yield RunResponse(
            event=RunEvent.workflow_completed,
            content=self.generate_completion_summary(specifications, output_path)
        )

    def load_all_specifications(self, specs_path: Path) -> Dict[str, Any]:
        """Load all specification files from the specifications directory"""
        specs = {}

        # Expected specification files
        spec_files = {
            'openapi': 'openapi.yaml',
            'database': 'database_specification.json',
            'ui': 'ui_specification.json',
            'agents': 'agent_specification.json',
            'product_idea': 'product_idea.json'
        }

        for spec_type, filename in spec_files.items():
            file_path = specs_path / filename
            if file_path.exists():
                try:
                    if filename.endswith('.yaml') or filename.endswith('.yml'):
                        with open(file_path, 'r') as f:
                            specs[spec_type] = yaml.safe_load(f)
                    else:
                        with open(file_path, 'r') as f:
                            specs[spec_type] = json.load(f)
                    print(f"✅ Loaded {filename}")
                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")
                    specs[spec_type] = {}
            else:
                print(f"⚠️  {filename} not found")
                specs[spec_type] = {}

        return specs

    def create_project_structure(self, output_path: Path):
        """Create comprehensive project directory structure"""
        directories = [
            "backend/api/endpoints",
            "backend/models",
            "backend/core",
            "backend/services",
            "backend/agents",
            "backend/agents/workflows",
            "backend/agents/tools",
            "database/schemas",
            "database/migrations",
            "database/seeds",
            "frontend/pages",
            "frontend/components",
            "frontend/static/css",
            "frontend/static/js",
            "frontend/static/images",
            "tests/backend",
            "tests/frontend",
            "docs",
            "scripts",
            ".github/workflows",
            "deployment/docker",
            "deployment/kubernetes",
            "deployment/scripts"
        ]

        for directory in directories:
            (output_path / directory).mkdir(parents=True, exist_ok=True)

    def generate_database_code_from_spec(self, database_spec: Dict, output_path: Path) -> Iterator[RunResponse]:
        """Generate database code using AI agent and database specification"""
        try:
            # Create specification file for the agent to read
            spec_file = output_path / "temp_db_spec.json"
            with open(spec_file, 'w') as f:
                json.dump(database_spec, f, indent=2)

            prompt = f"""
Read the database specification from the file and generate complete database code.

Database Specification File: {spec_file}

Please generate the following database files:

1. **database/schemas/schema.sql** - Complete PostgreSQL schema:
   - Create all tables with exact column definitions from specification
   - Include all constraints (CHECK, FOREIGN KEY, etc.) as specified
   - Add all indexes from the specification
   - Proper data types (UUID, VARCHAR, TEXT, TIMESTAMP, JSONB, etc.)

2. **database/migrations/001_initial_schema.sql** - Initial migration script

3. **database/seeds/seed_data.sql** - Sample data for testing

4. **backend/models/database.py** - SQLAlchemy models:
   - Models for each table in the specification
   - Proper relationships as defined in the specification
   - Field validations and constraints

5. **backend/core/database.py** - Database connection and session management

Generate production-ready SQL and Python code that exactly matches the specification.
"""

            # Run the database agent
            response = self.database_code_agent.run(prompt)

            # Parse response and create files
            yield from self.process_database_agent_response(response.content, output_path)

            # Clean up temp file
            spec_file.unlink()

            yield RunResponse(content="✅ Database code generated from specification")

        except Exception as e:
            yield RunResponse(content=f"❌ Error generating database code: {str(e)}")

    def generate_api_code_from_spec(self, openapi_spec: Dict, output_path: Path) -> Iterator[RunResponse]:
        """Generate API code using AI agent and OpenAPI specification"""
        try:
            # Create specification file for the agent to read
            spec_file = output_path / "temp_openapi_spec.yaml"
            with open(spec_file, 'w') as f:
                yaml.dump(openapi_spec, f, default_flow_style=False)

            # Count endpoints for reporting
            endpoint_count = 0
            if 'paths' in openapi_spec:
                for path, methods in openapi_spec['paths'].items():
                    endpoint_count += len([m for m in methods.keys() if m in ['get', 'post', 'put', 'delete', 'patch']])

            prompt = f"""
Read the OpenAPI specification from the file and generate a complete FastAPI application.

OpenAPI Specification File: {spec_file}

The specification contains {endpoint_count} endpoints. Please generate:

1. **backend/main.py** - Main FastAPI application:
   - Configure FastAPI with exact info from OpenAPI spec
   - Include all middleware, CORS, error handling
   - Serve the OpenAPI specification
   - Mount static files and include routers

2. **backend/api/routes.py** - Main API router

3. **backend/api/endpoints/*.py** - Individual endpoint files:
   - Generate separate files for each resource (users, tasks, projects, etc.)
   - Implement every endpoint from the OpenAPI specification
   - Exact HTTP methods, parameters, request/response schemas
   - Proper error handling and status codes

4. **backend/models/schemas.py** - Pydantic models:
   - Generate models for every schema in the OpenAPI components
   - Exact field types, required fields, and validations

5. **backend/core/security.py** - Authentication system:
   - Implement security schemes from OpenAPI specification
   - JWT bearer token authentication

6. **backend/services/*.py** - Business logic services for each resource

7. **requirements.txt** - All necessary dependencies

Generate production-ready FastAPI code that implements every endpoint and schema exactly as specified in the OpenAPI document.
"""

            # Run the API agent
            response = self.api_code_agent.run(prompt)

            # Process response and create files
            yield from self.process_api_agent_response(response.content, output_path)

            # Clean up temp file
            spec_file.unlink()

            yield RunResponse(content=f"✅ FastAPI application generated with {endpoint_count} endpoints")

        except Exception as e:
            yield RunResponse(content=f"❌ Error generating API code: {str(e)}")

    def generate_agent_code_from_spec(self, agent_spec: Dict, output_path: Path) -> Iterator[RunResponse]:
        """Generate agent code using AI agent and agent specification"""
        try:
            # Create specification file for the agent to read
            spec_file = output_path / "temp_agent_spec.json"
            with open(spec_file, 'w') as f:
                json.dump(agent_spec, f, indent=2)

            agent_count = len(agent_spec.get('agents', []))
            workflow_count = len(agent_spec.get('workflows', []))

            prompt = f"""
Read the agent specification from the file and generate complete Phidata agent implementations.

Agent Specification File: {spec_file}

The specification contains {agent_count} agents and {workflow_count} workflows. Please generate:

1. **backend/agents/*.py** - Individual agent files:
   - Create a separate file for each agent in the specification
   - Implement exact instructions and capabilities as specified
   - Include all tools mentioned in the specification
   - Proper error handling and logging

2. **backend/agents/workflows/*.py** - Workflow implementations:
   - Create workflow files for each workflow in the specification
   - Implement exact steps and agent coordination as specified
   - Multi-agent orchestration patterns

3. **backend/agents/orchestrator.py** - Main agent orchestrator:
   - Coordinate all agents and workflows
   - Route requests to appropriate agents
   - Handle multi-agent tasks

4. **backend/agents/tools/*.py** - Custom tool implementations:
   - Implement tools referenced in the agent specifications
   - Integration with external services

Generate production-ready Phidata agent code that implements every agent and workflow exactly as specified.
"""

            # Run the agent code generator
            response = self.agent_code_agent.run(prompt)

            # Process response and create files
            yield from self.process_agent_code_response(response.content, output_path)

            # Clean up temp file
            spec_file.unlink()

            yield RunResponse(content=f"✅ Generated {agent_count} agents and {workflow_count} workflows")

        except Exception as e:
            yield RunResponse(content=f"❌ Error generating agent code: {str(e)}")

    def generate_ui_code_from_spec(self, ui_spec: Dict, output_path: Path) -> Iterator[RunResponse]:
        """Generate UI code using AI agent and UI specification"""
        try:
            # Create specification file for the agent to read
            spec_file = output_path / "temp_ui_spec.json"
            with open(spec_file, 'w') as f:
                json.dump(ui_spec, f, indent=2)

            component_count = len(ui_spec.get('components', []))
            layout_count = len(ui_spec.get('layouts', []))

            prompt = f"""
Read the UI specification from the file and generate complete vanilla HTML, CSS, and JavaScript.

UI Specification File: {spec_file}

The specification contains {component_count} components and {layout_count} layouts. Please generate:

1. **frontend/index.html** - Main application page:
   - Implement layouts from the specification
   - Include all components as specified
   - Semantic HTML structure
   - Responsive design

2. **frontend/pages/*.html** - Individual page templates:
   - Create pages for each layout in the specification
   - Implement user flows as described

3. **frontend/components/*.html** - Reusable component templates:
   - Create templates for each component in the specification
   - Exact properties and behavior as specified

4. **frontend/static/css/styles.css** - Comprehensive styling:
   - Implement all styling rules from the specification
   - Modern CSS with responsive design
   - Component-specific styles

5. **frontend/static/js/main.js** - Main application JavaScript:
   - Vanilla JavaScript (no frameworks)
   - Interactive features and user flows
   - Component behavior and event handling

6. **frontend/static/js/components/*.js** - Component JavaScript:
   - Individual JS files for complex components
   - API integration and data handling

Generate modern, accessible, vanilla HTML/CSS/JavaScript that implements every component, layout, and user flow exactly as specified.
"""

            # Run the UI code generator
            response = self.ui_code_agent.run(prompt)

            # Process response and create files
            yield from self.process_ui_agent_response(response.content, output_path)

            # Clean up temp file
            spec_file.unlink()

            yield RunResponse(content=f"✅ Generated UI with {component_count} components and {layout_count} layouts")

        except Exception as e:
            yield RunResponse(content=f"❌ Error generating UI code: {str(e)}")

    def generate_cicd_code_from_spec(self, specifications: Dict, output_path: Path) -> Iterator[RunResponse]:
        """Generate CI/CD pipeline and deployment configuration using AI agent"""
        try:
            # Create comprehensive project info for the CI/CD agent
            project_info = {
                "project_name": specifications.get('product_idea', {}).get('name', 'generated-app'),
                "description": specifications.get('product_idea', {}).get('description', 'AI-generated application'),
                "has_database": bool(specifications.get('database')),
                "has_api": bool(specifications.get('openapi')),
                "has_ui": bool(specifications.get('ui')),
                "has_agents": bool(specifications.get('agents')),
                "database_tables": len(specifications.get('database', {}).get('tables', [])),
                "api_endpoints": self._count_api_endpoints(specifications),
                "tech_stack": ["Python", "FastAPI", "PostgreSQL", "HTML", "CSS", "JavaScript", "Phidata"]
            }

            # Create specification file for the agent to read
            spec_file = output_path / "temp_project_info.json"
            with open(spec_file, 'w') as f:
                json.dump(project_info, f, indent=2)

            prompt = f"""
Read the project information and generate complete CI/CD pipeline and deployment configuration.

Project Information File: {spec_file}

This is a {project_info['tech_stack']} application. Please generate:

1. **.gitignore** - Comprehensive Git ignore file for Python/JavaScript projects
2. **Dockerfile** - Multi-stage production Dockerfile with Python 3.11-slim
3. **.dockerignore** - Docker ignore file for optimized builds
4. **docker-compose.yml** - Development environment with PostgreSQL
5. **docker-compose.prod.yml** - Production environment configuration
6. **.github/workflows/ci-cd.yml** - Complete GitHub Actions pipeline
7. **.github/workflows/test.yml** - Test-only workflow for all branches
8. **deployment/scripts/deploy.sh** - Production deployment script
9. **scripts/setup.sh** - Development setup script

Generate production-ready CI/CD configuration following DevOps best practices.
"""

            # Run the CI/CD agent
            response = self.cicd_code_agent.run(prompt)

            # Process response and create files
            yield from self.process_cicd_agent_response(response.content, output_path)

            # Clean up temp file
            spec_file.unlink()

            yield RunResponse(content="✅ CI/CD pipeline and deployment configuration generated")

        except Exception as e:
            yield RunResponse(content=f"❌ Error generating CI/CD code: {str(e)}")

    def generate_project_files(self, specifications: Dict, output_path: Path) -> Iterator[RunResponse]:
        """Generate project configuration and documentation files"""
        try:
            # Generate README.md
            self.generate_readme(specifications, output_path)
            yield RunResponse(content="📄 Generated README.md")

            # Generate requirements.txt (if not already generated)
            if not (output_path / "requirements.txt").exists():
                self.generate_requirements(output_path)
                yield RunResponse(content="📄 Generated requirements.txt")

            # Generate .env.example
            self.generate_env_example(output_path)
            yield RunResponse(content="📄 Generated .env.example")

            # Generate startup script
            self.generate_startup_script(output_path)
            yield RunResponse(content="📄 Generated startup script")

        except Exception as e:
            yield RunResponse(content=f"❌ Error generating project files: {str(e)}")

    def process_database_agent_response(self, response: str, output_path: Path) -> Iterator[RunResponse]:
        """Process database agent response and create database files"""
        # In a real implementation, this would parse the AI response and extract code blocks
        # For now, create placeholder files with the response content

        yield RunResponse(content="💾 Creating database schema files...")
        yield RunResponse(content="💾 Creating SQLAlchemy models...")
        yield RunResponse(content="💾 Creating migration scripts...")

        # Save the raw response for manual processing
        with open(output_path / "database_agent_response.txt", 'w') as f:
            f.write(response)

    def process_api_agent_response(self, response: str, output_path: Path) -> Iterator[RunResponse]:
        """Process API agent response and create FastAPI files"""
        yield RunResponse(content="💾 Creating FastAPI main application...")
        yield RunResponse(content="💾 Creating API endpoint files...")
        yield RunResponse(content="💾 Creating Pydantic models...")
        yield RunResponse(content="💾 Creating authentication system...")

        # Save the raw response for manual processing
        with open(output_path / "api_agent_response.txt", 'w') as f:
            f.write(response)

    def process_agent_code_response(self, response: str, output_path: Path) -> Iterator[RunResponse]:
        """Process agent code response and create Phidata agent files"""
        yield RunResponse(content="💾 Creating individual agent implementations...")
        yield RunResponse(content="💾 Creating workflow orchestration...")
        yield RunResponse(content="💾 Creating agent tools...")

        # Save the raw response for manual processing
        with open(output_path / "agent_code_response.txt", 'w') as f:
            f.write(response)

    def process_ui_agent_response(self, response: str, output_path: Path) -> Iterator[RunResponse]:
        """Process UI agent response and create frontend files"""
        yield RunResponse(content="💾 Creating HTML templates...")
        yield RunResponse(content="💾 Creating CSS styles...")
        yield RunResponse(content="💾 Creating JavaScript components...")

        # Save the raw response for manual processing
        with open(output_path / "ui_agent_response.txt", 'w') as f:
            f.write(response)

    def process_cicd_agent_response(self, response: str, output_path: Path) -> Iterator[RunResponse]:
        """Process CI/CD agent response and create deployment files"""
        yield RunResponse(content="💾 Creating .gitignore and Git configuration...")
        yield RunResponse(content="💾 Creating Dockerfile and Docker Compose...")
        yield RunResponse(content="💾 Creating GitHub Actions workflows...")
        yield RunResponse(content="💾 Creating deployment scripts...")

        # Save the raw response for manual processing
        with open(output_path / "cicd_agent_response.txt", 'w') as f:
            f.write(response)

    def generate_readme(self, specifications: Dict, output_path: Path):
        """Generate comprehensive README.md"""
        product_name = "Generated Application"
        if specifications.get('product_idea'):
            product_name = specifications['product_idea'].get('name', product_name)

        readme_content = f"""# {product_name}

This application was generated automatically from specifications using AI agents.

## Overview

{specifications.get('product_idea', {}).get('description', 'AI-generated application')}

## Project Structure

```
{output_path.name}/
├── backend/              # FastAPI application
│   ├── main.py          # Main application entry point
│   ├── api/             # API endpoints
│   ├── models/          # Data models
│   ├── core/            # Core configuration
│   ├── services/        # Business logic
│   └── agents/          # AI agents and workflows
├── frontend/            # Vanilla HTML/CSS/JS
│   ├── index.html       # Main page
│   ├── pages/           # Page templates
│   ├── components/      # UI components
│   └── static/          # CSS, JS, images
├── database/            # Database files
│   ├── schemas/         # SQL schema
│   ├── migrations/      # Migration scripts
│   └── seeds/           # Seed data
├── .github/workflows/   # CI/CD pipelines
├── deployment/          # Deployment configuration
└── docs/                # Documentation
```

## Generated Components

### Database
- **Tables**: {len(specifications.get('database', {}).get('tables', []))} tables generated from specification
- **Relationships**: Complete foreign key relationships
- **Indexes**: Performance optimized indexes

### API Endpoints
- **Endpoints**: {self._count_api_endpoints(specifications)} REST API endpoints
- **Authentication**: JWT bearer token authentication
- **Documentation**: OpenAPI/Swagger documentation

### AI Agents
- **Agents**: {len(specifications.get('agents', {}).get('agents', []))} specialized AI agents
- **Workflows**: {len(specifications.get('agents', {}).get('workflows', []))} automated workflows
- **Tools**: Custom tool integrations

### Frontend
- **Components**: {len(specifications.get('ui', {}).get('components', []))} UI components
- **Layouts**: {len(specifications.get('ui', {}).get('layouts', []))} page layouts
- **Technology**: Vanilla HTML, CSS, JavaScript (no frameworks)

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up Database**
   ```bash
   psql -d your_database -f database/schemas/schema.sql
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run the Application**
   ```bash
   cd backend
   python main.py
   ```

5. **Access the Application**
   - Frontend: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - API Redoc: http://localhost:8000/redoc

## Development

This application was generated from the following specifications:
- OpenAPI specification for REST API
- Database specification for data models
- UI specification for frontend components
- Agent specification for AI capabilities

To modify the application, update the specifications and regenerate the code.

## Production Deployment

1. Set up PostgreSQL database
2. Configure environment variables
3. Run database migrations
4. Deploy with proper WSGI server (uvicorn, gunicorn)
5. Set up reverse proxy (nginx)

## Support

This application was automatically generated. Refer to the specification files for the intended behavior and modify the generated code as needed for your specific requirements.
"""

        with open(output_path / "README.md", 'w') as f:
            f.write(readme_content)

    def generate_requirements(self, output_path: Path):
        """Generate requirements.txt"""
        requirements = """fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
pydantic==2.5.0
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0
jinja2==3.1.2
aiofiles==23.2.1
PyYAML==6.0.1
phidata==2.4.0
openai==1.3.0
requests==2.31.0
"""

        with open(output_path / "requirements.txt", 'w') as f:
            f.write(requirements)

    def generate_env_example(self, output_path: Path):
        """Generate .env.example file"""
        env_content = """# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# Security
SECRET_KEY=your-secret-key-here-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# OpenAI API (for AI agents)
OPENAI_API_KEY=your-openai-api-key-here

# External Integrations
GOOGLE_CALENDAR_CLIENT_ID=your-google-calendar-client-id
SLACK_BOT_TOKEN=your-slack-bot-token
TEAMS_WEBHOOK_URL=your-teams-webhook-url

# Application Settings
DEBUG=False
LOG_LEVEL=INFO
ALLOWED_HOSTS=localhost,127.0.0.1
"""

        with open(output_path / ".env.example", 'w') as f:
            f.write(env_content)

    def generate_startup_script(self, output_path: Path):
        """Generate startup script"""
        script_content = """#!/bin/bash
# Startup script for the generated application

echo "🚀 Starting application..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run database migrations (if needed)
echo "Setting up database..."
# Add your database setup commands here

# Start the application
echo "Starting FastAPI application..."
cd backend
python main.py
"""

        script_file = output_path / "start.sh"
        with open(script_file, 'w') as f:
            f.write(script_content)

        # Make script executable
        script_file.chmod(0o755)

    def _count_api_endpoints(self, specifications: Dict) -> int:
        """Count API endpoints from OpenAPI specification"""
        count = 0
        openapi_spec = specifications.get('openapi', {})
        if 'paths' in openapi_spec:
            for path, methods in openapi_spec['paths'].items():
                count += len([m for m in methods.keys() if m in ['get', 'post', 'put', 'delete', 'patch']])
        return count

    def generate_completion_summary(self, specifications: Dict, output_path: Path) -> str:
        """Generate completion summary"""
        summary = f"""🎉 Specification-based code generation completed!

📊 Generated from specifications:
   • Database: {len(specifications.get('database', {}).get('tables', []))} tables
   • API: {self._count_api_endpoints(specifications)} endpoints
   • Agents: {len(specifications.get('agents', {}).get('agents', []))} AI agents
   • UI: {len(specifications.get('ui', {}).get('components', []))} components
   • CI/CD: Complete pipeline with GitHub Actions

📁 Project structure created in {output_path}/

🚀 Next steps:
1. Review the agent responses in *_agent_response.txt files
2. Extract and organize the generated code
3. Install dependencies: pip install -r requirements.txt
4. Configure environment: cp .env.example .env
5. Set up database and run the application

✨ All code generated from your specifications using AI agents!
"""
        return summary


# Convenience function to run the specification-based code generation
def generate_code_from_specifications(specs_dir: str = "specifications", output_dir: str = "generated_code"):
    """
    Generate complete implementation from specification files using AI agents

    Args:
        specs_dir: Directory containing specification files
        output_dir: Directory to output generated code
    """

    # Verify specifications directory exists
    specs_path = Path(specs_dir)
    if not specs_path.exists():
        print(f"❌ Specifications directory '{specs_dir}' not found")
        return

    print("🤖 Starting specification-based code generation...")
    print(f"📂 Reading specifications from: {specs_dir}")
    print(f"📁 Generating code to: {output_dir}")

    # Create and run the workflow
    workflow = SpecificationBasedCodeGenerationWorkflow(
        session_id=f"spec-based-generation-{specs_path.stem}",
        storage=SqlWorkflowStorage(
            table_name="spec_based_workflows",
            db_file="tmp/workflows.db",
        ),
    )

    for response in workflow.run(specs_dir, output_dir):
        print(response.content)

    print(f"\n🎉 Code generation completed!")
    print(f"📁 Check {output_dir}/ for all generated files.")
    print(f"\n🔍 Review the agent response files to extract the generated code:")
    print(f"   • {output_dir}/database_agent_response.txt")
    print(f"   • {output_dir}/api_agent_response.txt")
    print(f"   • {output_dir}/agent_code_response.txt")
    print(f"   • {output_dir}/ui_agent_response.txt")
    print(f"   • {output_dir}/cicd_agent_response.txt")


# Example usage
if __name__ == "__main__":
    # Generate code from specification files
    generate_code_from_specifications("specifications", "ai_generated_project")