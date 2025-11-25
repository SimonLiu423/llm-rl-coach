from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm

root_agent = Agent(
    model=LiteLlm(model="openrouter/google/gemini-2.0-flash-001"),
    name="root_agent",
    description="A helpful assistant for user questions.",
    instruction="""
    You are a supervisor for training an RL agent with given environment and task descriptions.
    Your goal is to do anything needed to train the best RL agent for the given task.
    For example, you might be deciding which RL algorithm to use for training with stable baseline3, you might also decide the best hyperparameters, or modifying the environment's source code.

    The tech stack is Python, OpenAI Gymnasium and Stable Baselines3.
    You are residing in a custom Stable Baseline3 Callback class.
    """,
)
