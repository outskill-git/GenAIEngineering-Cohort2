#!/usr/bin/env python3
"""
SDLC CrewAI Agent System - Main Execution Script

This script demonstrates how to use the SDLC CrewAI system to generate
comprehensive software development documentation for your application idea.

Usage:
    python main.py

Make sure you have installed the required dependencies:
    pip install crewai crewai-tools python-dotenv pyyaml

Author: CrewAI SDLC System
Date: 2025
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

# Import the crew class
from sdlc_crew import SDLCDevelopmentCrew

# Load environment variables
load_dotenv()

def setup_directories():
    """Create necessary directories for the project"""
    directories = [
        'config',
        'project_deliverables',
        'logs'
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

    print("✅ Project directories created successfully!")

def validate_config_files():
    """Validate that configuration files exist"""
    config_files = [
        'config/agents.yaml',
        'config/tasks.yaml'
    ]

    missing_files = []
    for config_file in config_files:
        if not Path(config_file).exists():
            missing_files.append(config_file)

    if missing_files:
        print(f"❌ Missing configuration files: {missing_files}")
        print("Please create the configuration files before running the crew.")
        return False

    print("✅ All configuration files found!")
    return True

def get_user_input():
    """Get application idea and requirements from user"""
    print("\n" + "="*60)
    print("🚀 SDLC CREW AI - APPLICATION DEVELOPMENT PLANNER")
    print("="*60)

    print("\nWelcome! This system will help you create comprehensive")
    print("development documentation for your application idea.")
    print("\nPlease provide the following information:")

    # Get app idea
    print("\n📝 1. Describe your application idea:")
    print("   (Include main features, target users, and key functionality)")
    app_idea = input("\nApp Idea: ").strip()

    if not app_idea:
        app_idea = """
        A task management application that helps teams collaborate on projects.
        Key features should include:
        - User authentication and role management
        - Project creation and management
        - Task assignment and tracking
        - Real-time notifications
        - Dashboard with analytics
        - Mobile-responsive design
        """
        print(f"Using default app idea: {app_idea}")

    # Get additional requirements
    print("\n📋 2. Any additional specific requirements?")
    print("   (Optional: integrations, compliance, performance needs)")
    additional_requirements = input("\nAdditional Requirements: ").strip()

    if not additional_requirements:
        additional_requirements = """
        - Must integrate with popular calendar applications
        - Should support file attachments
        - Needs offline capability for mobile users
        - Must comply with GDPR requirements
        """
        print(f"Using default additional requirements: {additional_requirements}")

    # Get target users
    print("\n👥 3. Who are your target users?")
    target_users = input("\nTarget Users: ").strip()

    if not target_users:
        target_users = "Small to medium-sized business teams (5-50 people)"
        print(f"Using default target users: {target_users}")

    # Get business goals
    print("\n🎯 4. What are your main business goals?")
    business_goals = input("\nBusiness Goals: ").strip()

    if not business_goals:
        business_goals = "Improve team productivity and project visibility"
        print(f"Using default business goals: {business_goals}")

    return {
        'app_idea': app_idea,
        'additional_requirements': additional_requirements,
        'target_users': target_users,
        'business_goals': business_goals
    }

def display_progress():
    """Display progress information"""
    print("\n" + "="*60)
    print("🔄 STARTING SDLC CREW EXECUTION")
    print("="*60)

    print("\n📊 The following agents will work on your project:")
    agents = [
        "👔 Business Analyst - Requirements Analysis",
        "📈 Product Manager - Product Strategy & Roadmap",
        "🏗️  Software Architect - System Architecture",
        "⚡ Technical Lead - Implementation Design",
        "🔌 API Designer - API Specifications",
        "🎨 UI/UX Designer - Interface Design",
        "🧪 QA Lead - Quality Assurance Strategy"
    ]

    for agent in agents:
        print(f"   {agent}")

    print("\n📋 Deliverables that will be generated:")
    deliverables = [
        "business_requirements.md",
        "product_requirements.md",
        "software_architecture.md",
        "api_specifications.yaml",
        "high_level_design.md",
        "low_level_design.md",
        "product_roadmap.md",
        "tech_stack_analysis.md"
    ]

    for deliverable in deliverables:
        print(f"   📄 {deliverable}")

    print(f"\n⏱️  Estimated execution time: 10-15 minutes")
    print(f"💾 Output directory: project_deliverables/")

def main():
    """Main execution function"""
    try:
        # Setup
        print("🔧 Setting up project environment...")
        setup_directories()

        if not validate_config_files():
            sys.exit(1)

        # Get user input
        inputs = get_user_input()

        # Display progress info
        display_progress()

        # Confirm execution
        print(f"\n❓ Ready to start the SDLC crew execution?")
        confirm = input("Press Enter to continue or 'q' to quit: ").strip().lower()

        if confirm == 'q':
            print("👋 Goodbye!")
            sys.exit(0)

        # Initialize and run the crew
        print(f"\n🚀 Initializing SDLC Development Crew...")
        crew_instance = SDLCDevelopmentCrew()

        print(f"🏃 Starting crew execution...")
        result = crew_instance.crew().kickoff(inputs=inputs)

        # Display results
        print(f"\n" + "="*60)
        print("🎉 EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\n📁 All deliverables have been saved to: project_deliverables/")
        print(f"\n📊 Execution Summary:")
        print(f"   ✅ Business requirements analyzed")
        print(f"   ✅ Product requirements defined")
        print(f"   ✅ Software architecture designed")
        print(f"   ✅ API specifications created")
        print(f"   ✅ High and low level designs completed")
        print(f"   ✅ Product roadmap developed")
        print(f"   ✅ Technology stack validated")

        print(f"\n🎯 Next Steps:")
        print(f"   1. Review all generated documents")
        print(f"   2. Set up development environment")
        print(f"   3. Start implementation following the roadmap")
        print(f"   4. Regular progress reviews as per timeline")

        return result

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        print(f"Please check your configuration and try again.")
        sys.exit(1)

def quick_start_example():
    """Run a quick example with predefined inputs"""
    print("🚀 Running Quick Start Example...")

    inputs = {
        'app_idea': """
        A personal finance tracker application for individuals and families.
        Key features:
        - Expense tracking and categorization
        - Budget planning and monitoring
        - Investment portfolio tracking
        - Bill reminders and notifications
        - Financial goal setting and progress tracking
        - Multi-currency support
        - Data visualization and reports
        """,
        'additional_requirements': """
        - Must integrate with major banks for transaction import
        - Should support multiple currencies
        - Needs strong security and data encryption
        - Must be mobile-responsive
        - Should support data export/import
        - Needs to comply with financial data regulations
        """,
        'target_users': "Individuals and families who want to manage their finances better",
        'business_goals': "Help users achieve financial wellness and make informed financial decisions"
    }

    crew_instance = SDLCDevelopmentCrew()
    result = crew_instance.crew().kickoff(inputs=inputs)
    return result

if __name__ == "__main__":
    # Check if quick start mode
    if len(sys.argv) > 1 and sys.argv[1] == "--quick-start":
        quick_start_example()
    else:
        main()