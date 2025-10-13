from langchain.chat_models import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents import Tool

# Define a tool the agent can use
tools = [
    Tool(
        name="Calculator",
        func=lambda x: str(eval(x)),
        description="Useful for doing math calculations"
    )
]

# Create a chat model
llm = ChatOpenAI(temperature=0)

# Create a ReAct agent
agent = create_react_agent(llm, tools)

# Wrap in an executor to run
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run agent
executor.run("What is 23 * 17?")
