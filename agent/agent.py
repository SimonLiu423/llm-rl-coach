import asyncio

import gymnasium as gym
import optuna
from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner, Runner
from google.genai.types import Content, Part
from optuna.samplers import BaseSampler
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from .wrapper import AgentModifiable


class RLCoach:
    def __init__(self, env_id: str, env_desc: str, n_envs: int = 4):
        self.env_id = env_id
        self.env_desc = env_desc
        self.n_envs = n_envs
        self.avail_algo = {"A2C": A2C, "DQN": DQN, "PPO": PPO, "SAC": SAC, "TD3": TD3}
        # To be decided by agent
        self.algo = None
        self.step_code = None
        # Session constants for the ADK runner
        self.APP_NAME = "agents"
        self.USER_ID = "user"
        self.SESSION_ID = "session"

        self.agent = Agent(
            model=LiteLlm(
                model="openrouter/openai/gpt-5.1",
            ),
            name="RL_Coach",
            description="A helpful assistant for user questions.",
            instruction="""
            You are a supervisor for training an RL agent with given environment and task descriptions.
            Your goal is to do anything needed to train the best RL agent for the given task.
            For example, you might be deciding which RL algorithm to use for training with stable baseline3, you might also decide the best hyperparameters, or modifying the environment's source code.
        
            The tech stack is Python, OpenAI Gymnasium and Stable Baselines3.
            You are residing in a custom Stable Baseline3 Callback class.
            """,
            tools=[self._modify_env_step, self._select_algorithm],
        )
        self.runner = self._start_runner()
        self.session_service = self.runner.session_service

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

    def _modify_env_step(self, code_str: str) -> dict:
        """
        Modify the environment's gym.Wrapper `step` method with the provided code dynamically.
        The code is executed with `exec` and the wrapper's `step` method will be replaced with the
        modified version.

        Args:
            code_str (str): A string containing valid Python code defining a function
            `step(self, action)`. The function MUST START by calling `self.env.step(action)`
            since your code should act as a wrapper for the original `step` method.
            Any dependencies must be imported inside the scope of the function.

        Returns:
            dict: Success status or error message.
        """
        print("Tool called: modify_step_fn")
        self.step_code = code_str
        print(code_str)
        # ret = self.train_vec_env.env_method("modify_step_fn", code_str)
        # return ret[0]
        return {"success": True}

    def _select_algorithm(self, algo_name: str) -> dict:
        """
        Select a RL algorithm to train this agent.

        Args:
            algo_name (str): The name of the algorithm to select.

        Returns:
            dict: Success status or error message.
        """
        try:
            self.algo = self.avail_algo[algo_name]
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def query(self, prompt: str):
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

    def query_reward(self):
        env = gym.make(self.env_id)
        env = AgentModifiable(env)

        self.query(
            prompt=f"""
                    A training session is about to start. Here's the description of the environment:
                    <env_description>
                    {self.env_desc}
                    </env_description>

                    And here's some information about the current `gym.Env` instance:
                    <env_info>
                    {env.env_info()}
                    </env_info>

                    Modify the environment's `step` method to implement a custom reward function.
                    You should use `_modify_env_step` tool to modify the `step` method.
                    """
        )

    def query_algorithm(self):
        self.query(
            prompt=f"""
                    Select a Stable Baselines3 algorithm to train this agent.
                    Available algorithms: {self.avail_algo.keys()}
                    """
        )

    def objective(self, trial: optuna.Trial):
        # Suggest hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        gamma = trial.suggest_float("gamma", 0.9, 0.9999)
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)

        # Define model parameters
        model_params = {
            "policy": "MlpPolicy",
            "n_steps": 4096,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
        }

        eval_env = gym.make(self.env_id, render_mode="rgb_array")
        eval_env = AgentModifiable(eval_env, self.step_code)
        eval_env = Monitor(eval_env)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="best_model",
            log_path=None,
            verbose=1,
        )

        wrapper_kwargs = {"step_code": self.step_code}

        vec_env = make_vec_env(
            self.env_id,
            n_envs=self.n_envs,
            wrapper_class=AgentModifiable,
            wrapper_kwargs=wrapper_kwargs,
        )
        model = self.algo(env=vec_env, verbose=0, **model_params)

        coach_callback = LLMCoachCallback(coach=self, env_desc=self.env_desc)
        callbacks = [eval_callback, coach_callback]

        model.learn(total_timesteps=250_000, callback=callbacks)

        mean_reward = eval_callback.best_mean_reward

        return mean_reward

    def optimize(self):
        # sampler = LLMSampler(runner=self.runner)
        self.query_reward()
        self.query_algorithm()

        assert self.algo is not None, "Algorithm not selected"
        assert self.step_code is not None, "Step code not modified"

        study = optuna.create_study(
            direction="maximize",
            storage="sqlite:///db.sqlite3",  # Saves data to file for dashboard
            study_name="rl_coach_session",
        )
        study.optimize(self.objective, n_trials=10, show_progress_bar=True)

    def run_best_model(self):
        best_model_path = "./best_model/best_model.zip"
        model = PPO.load(best_model_path)

        eval_env = gym.make(self.env_id, render_mode="human")
        obs, _ = eval_env.reset()

        while True:
            action, _states = model.predict(obs, deterministic=True)

            obs, rewards, terminated, truncated, info = eval_env.step(action.item())

            if terminated or truncated:
                obs, _ = eval_env.reset()


class LLMCoachCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(
        self, coach: RLCoach, env_desc: str, check_freq: int = 100, verbose: int = 0
    ):
        super().__init__(verbose)
        self.coach = coach
        self.check_freq = check_freq
        self.env_desc = env_desc

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # self.coach.query(
        #     prompt=f"""
        #             A training session is about to start. Here's the description of the environment:
        #             <env_description>
        #             {self.env_desc}
        #             </env_description>

        #             And here's some information about the current `gym.Env` instance:
        #             <env_info>
        #             {self._env_info()}
        #             </env_info>

        #             Modify the environment's `step` method to implement a custom reward function.
        #             """
        # )
        # print(self._env_info())
        pass

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


class LLMSampler(BaseSampler):
    def __init__(self, runner: Runner):
        self.runner = runner
        self._rng = optuna.random.RandomSampler()

    def infer_relative_search_space(self, study, trial):
        # Allow Optuna to handle search space definitions
        return optuna.search_space.intersection_search_space(
            study.get_trials(deepcopy=False)
        )

    def sample_relative(self, study, trial, search_space):
        # 1. Get history of previous trials (params + reward)
        history = ""
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                history += f"Params: {t.params}, Reward: {t.value}\n"

        # 2. Prompt the LLM
        prompt = f"""
        Here is the history of previous attempts to tune a PPO agent:
        {history}
        
        The search space is: {search_space}
        
        Suggest the next set of hyperparameters to maximize Reward. 
        Return ONLY a JSON string.
        """

        # 3. Call LLM (Pseudo-code)
        response = self.runner.chat.completions.create(
            messages=[{"role": "user", "content": prompt}]
        )
        # suggestion = parse_json(response)  # You'll need to write a parser

        # return suggestion

    def sample_independent(self, study, trial, param_name, param_distribution):
        # Fallback to random if LLM fails or for independent sampling
        return self._rng.sample_independent(
            study, trial, param_name, param_distribution
        )
