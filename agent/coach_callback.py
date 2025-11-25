import asyncio
import inspect

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner, Runner
from google.genai.types import Content, Part
from stable_baselines3.common.callbacks import BaseCallback


class LLMCoachCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(
        self, agent: Agent, env_desc: str, check_freq: int = 100, verbose: int = 0
    ):
        super().__init__(verbose)
        self.env_desc = env_desc
        self.check_freq = check_freq

        # Session constants for the ADK runner
        self.APP_NAME = "agents"
        self.USER_ID = "user"
        self.SESSION_ID = "session"

        self.agent = agent

        # CRITICAL: Register the modification methods as Tools for the Agent.
        # This allows the LLM to output a tool call to invoke these python methods directly.
        self.agent.tools.append(self._modify_step_fn)

        self.runner = self._start_runner()

    def _start_runner(self) -> Runner:
        """
        Initializes the InMemoryRunner and establishes an asynchronous session.

        Returns:
            Runner: The initialized ADK runner ready to process queries.
        """
        runner = InMemoryRunner(
            agent=self.agent,
            app_name=self.APP_NAME,
        )
        session_service = runner.session_service

        # Create the session asynchronously strictly for initialization
        asyncio.run(
            session_service.create_session(
                app_name=self.APP_NAME,
                user_id=self.USER_ID,
                session_id=self.SESSION_ID,
            )
        )
        return runner

    def _query(self, prompt: str):
        """
        Sends a natural language query to the Agent and processes the response stream.

        This method handles the conversation loop. If the Agent decides to call
        a tool (e.g., modify_step_fn), the Runner handles the execution automatically
        during the `runner.run` loop.

        Args:
            query (str): The text prompt from the user or supervisor loop.
        """
        user_message = Content(role="user", parts=[Part.from_text(text=prompt)])
        full_response_text = ""

        # Iterate through the event stream from the agent
        for event in self.runner.run(
            user_id=self.USER_ID, session_id=self.SESSION_ID, new_message=user_message
        ):
            # Capture partial text updates for streaming output
            if (
                event.partial
                and event.content
                and event.content.parts
                and event.content.parts[0].text
            ):
                full_response_text += event.content.parts[0].text

            # Handle the final response completion
            if event.is_final_response():
                if (
                    event.content
                    and event.content.parts
                    and event.content.parts[0].text
                ):
                    full_response_text += (
                        event.content.parts[0].text if not event.partial else ""
                    )
                print(f"LLM: {full_response_text}")

    def _modify_step_fn(self, code_str: str) -> dict:
        """
        Modify the environment's `step` method with the provided code dynamically.
        The code is executed with `exec` and the `step` method will be replaced with the
        modified version.

        Args:
            code_str (str): A string containing valid Python code defining a function
                            `step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`.

        Returns:
            dict: Success status or error message.
        """
        ret = self.training_env.env_method("modify_step_fn", code_str)
        return ret[0]

    def _env_info(self) -> dict:
        """
        Retrieves the current state and source code of the environment.

        This is intended to be passed to the Agent so it understands the current
        logic before attempting to modify it.

        Returns:
            dict: A dictionary containing spaces, metadata, and the raw source code
                  of the current `step` and `reset` methods.
        """
        return {
            "action_space": self.training_env.get_attr("action_space")[0].__dict__,
            "observation_space": self.training_env.get_attr("observation_space")[
                0
            ].__dict__,
            "spec": self.training_env.get_attr("spec")[0].__dict__,
            "metadata": self.training_env.get_attr("metadata")[0],
            # inspect.getsource lets the LLM read the actual Python implementation
            "step_fn": inspect.getsource(self.training_env.get_attr("step")[0]),
            "reset_fn": inspect.getsource(self.training_env.get_attr("reset")[0]),
        }

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self._query(
            prompt=f"""
                    A training session is about to start. Here's the description of the environment:
                    <env_description>
                    {self.env_desc}
                    </env_description>

                    And here's some information about the current `gym.Env` instance:
                    <env_info>
                    {self._env_info()}
                    </env_info>

                    Modify the environment's `step` method to implement a custom reward function.
                    """
        )
        print(self._env_info())

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        # if self.n_calls % self.check_freq == 0:
        # self._query()
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
