# LangChain Chain Types Tutorial
# This tutorial demonstrates Simple Chain, Sequential Chain, and Simple Sequential Chain

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain

# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=500
)

print("🚀 LangChain Chain Types Tutorial")
print("=" * 50)

# ============================================================================
# 1. SIMPLE CHAIN (Single Input/Output)
# ============================================================================
print("\n1️⃣ SIMPLE CHAIN")
print("-" * 20)

# Create a prompt template for generating a story
story_prompt = ChatPromptTemplate.from_template(
    "Write a short creative story (2-3 sentences) about {topic}. Make it interesting and engaging."
)

# Create a simple chain: Prompt → LLM → Output Parser
simple_chain = story_prompt | llm 


# Create a simple LLMChain
simple_chain = LLMChain(
    llm=llm,
    prompt=story_prompt,
)

# Execute the simple chain
topic = "a robot learning to paint"
story_result = simple_chain.invoke({"topic": topic})
print(f"Topic: {topic}")
print(f"Generated Story: {story_result}")

# ============================================================================
# 2. SIMPLE SEQUENTIAL CHAIN (Output of Chain 1 → Input of Chain 2)
# ============================================================================
print("\n2️⃣ SIMPLE SEQUENTIAL CHAIN")
print("-" * 25)

# Chain 1: Generate a business idea
idea_prompt = ChatPromptTemplate.from_template(
    "Generate a creative business idea for the industry: {industry}. "
    "Provide just the business idea in one sentence."
)
idea_chain = idea_prompt | llm | StrOutputParser()

# Chain 2: Create a marketing slogan for the business idea
slogan_prompt = ChatPromptTemplate.from_template(
    "Create a catchy marketing slogan for this business idea: {business_idea}. "
    "Make it memorable and under 10 words."
)
slogan_chain = slogan_prompt | llm | StrOutputParser()

# Create Simple Sequential Chain
# Output from idea_chain automatically becomes input for slogan_chain
simple_sequential_chain = SimpleSequentialChain(
    chains=[idea_chain, slogan_chain],
    verbose=True  # Shows intermediate outputs
)

# Execute the chain
industry = "sustainable technology"
final_result = simple_sequential_chain.invoke({"input": industry})
print(f"Industry: {industry}")
print(f"Final Marketing Slogan: {final_result['output']}")

# ============================================================================
# 3. SEQUENTIAL CHAIN (Multiple Inputs/Outputs with Named Variables)
# ============================================================================
print("\n3️⃣ SEQUENTIAL CHAIN")
print("-" * 18)

# Chain 1: Analyze a product concept
analysis_prompt = PromptTemplate(
    input_variables=["product_name", "target_market"],
    template="""
    Analyze this product concept:
    Product: {product_name}
    Target Market: {target_market}
    
    Provide a brief market analysis (2-3 sentences).
    """
)
analysis_chain = LLMChain(
    llm=llm,
    prompt=analysis_prompt,
    output_key="market_analysis"  # Named output
)

# Chain 2: Generate pricing strategy
pricing_prompt = PromptTemplate(
    input_variables=["product_name", "market_analysis"],
    template="""
    Based on this market analysis: {market_analysis}
    
    Suggest a pricing strategy for {product_name}.
    Include price range and reasoning (2-3 sentences).
    """
)
pricing_chain = LLMChain(
    llm=llm,
    prompt=pricing_prompt,
    output_key="pricing_strategy"  # Named output
)

# Chain 3: Create final business plan summary
business_plan_prompt = PromptTemplate(
    input_variables=["product_name", "target_market", "market_analysis", "pricing_strategy"],
    template="""
    Create a concise business plan summary using:
    
    Product: {product_name}
    Target Market: {target_market}
    Market Analysis: {market_analysis}
    Pricing Strategy: {pricing_strategy}
    
    Summarize in 3-4 sentences focusing on key opportunities.
    """
)
business_plan_chain = LLMChain(
    llm=llm,
    prompt=business_plan_prompt,
    output_key="business_plan"  # Named output
)

# Create Sequential Chain with multiple named inputs/outputs
sequential_chain = SequentialChain(
    chains=[analysis_chain, pricing_chain, business_plan_chain],
    input_variables=["product_name", "target_market"],  # Initial inputs
    output_variables=["market_analysis", "pricing_strategy", "business_plan"],  # All outputs
    verbose=True  # Shows all intermediate steps
)

# Execute the sequential chain
inputs = {
    "product_name": "Smart Fitness Mirror",
    "target_market": "health-conscious millennials"
}

sequential_result = sequential_chain.invoke(inputs)

print("📊 SEQUENTIAL CHAIN RESULTS:")
print(f"Product: {inputs['product_name']}")
print(f"Target Market: {inputs['target_market']}")
print(f"\n📈 Market Analysis:\n{sequential_result['market_analysis']}")
print(f"\n💰 Pricing Strategy:\n{sequential_result['pricing_strategy']}")
print(f"\n📋 Business Plan:\n{sequential_result['business_plan']}")

# ============================================================================
# SUMMARY OF DIFFERENCES
# ============================================================================
print("\n" + "=" * 60)
print("📚 SUMMARY OF CHAIN TYPES")
print("=" * 60)

print("""
1️⃣ SIMPLE CHAIN:
   • Single operation: Prompt → LLM → Output
   • Uses: prompt | llm | parser syntax
   • Best for: Individual tasks

2️⃣ SIMPLE SEQUENTIAL CHAIN:
   • Multiple chains in sequence
   • Output of Chain 1 → Input of Chain 2
   • Single input/output flow
   • Best for: Linear workflows

3️⃣ SEQUENTIAL CHAIN:
   • Multiple chains with named variables
   • Can handle multiple inputs/outputs
   • More complex data flow
   • Best for: Complex multi-step processes
""")

# ============================================================================
# PRACTICAL EXAMPLE: Content Creation Pipeline
# ============================================================================
print("\n🎯 PRACTICAL EXAMPLE: Blog Post Creation Pipeline")
print("-" * 50)

# Step 1: Generate blog topics
topic_generator = ChatPromptTemplate.from_template(
    "Generate 3 engaging blog post topics about {subject}. List them numbered 1-3."
) | llm | StrOutputParser()

# Step 2: Pick first topic and create outline
outline_generator = ChatPromptTemplate.from_template(
    "From these topics: {topics}\n\nPick the first topic and create a detailed outline with 4 main points."
) | llm | StrOutputParser()

# Step 3: Write introduction
intro_writer = ChatPromptTemplate.from_template(
    "Based on this outline: {outline}\n\nWrite an engaging introduction paragraph (3-4 sentences)."
) | llm | StrOutputParser()

# Create content pipeline
content_pipeline = SimpleSequentialChain(
    chains=[topic_generator, outline_generator, intro_writer],
    verbose=True
)

# Execute content creation pipeline
subject = "artificial intelligence in healthcare"
content_result = content_pipeline.invoke({"input": subject})

print(f"Subject: {subject}")
print(f"Final Blog Introduction:\n{content_result['output']}")

print("\n✅ Tutorial Complete! You've learned all three chain types in LangChain.")