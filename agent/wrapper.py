import inspect
import textwrap
import types

import gymnasium as gym


class AgentModifiable(gym.Wrapper):
    def __init__(self, env: gym.Env, step_code: str | None = None):
        super().__init__(env)
        if step_code:
            self.modify_wrap_step(step_code)

    def env_info(self) -> dict:
        """
        Retrieves the current state and source code of the environment.

        This is intended to be passed to the Agent so it understands the current
        logic before attempting to modify it.

        Returns:
            dict: A dictionary containing spaces, metadata, and the raw source code
                  of the current `step` and `reset` methods.
        """
        return {
            "action_space": self.action_space.__dict__,
            "observation_space": self.observation_space.__dict__,
            "spec": self.spec.__dict__,
            "metadata": self.metadata,
            # inspect.getsource lets the LLM read the actual Python implementation
            "step_fn": inspect.getsource(self.unwrapped.__class__.step),
            "reset_fn": inspect.getsource(self.unwrapped.__class__.reset),
        }

    def modify_wrap_step(self, code_str: str) -> dict:
        local_scope = {}
        clean_code = textwrap.dedent(code_str)
        try:
            exec(clean_code, globals=globals(), locals=local_scope)

            if "step" in local_scope:
                bound_method = types.MethodType(local_scope["step"], self)
                self.step = bound_method
                return {"success": True}
            else:
                return {
                    "success": False,
                    "error": "step function not found in provided code",
                }
        except Exception as e:
            print(f"Error in modify_step_fn: {e}")
            return {"success": False, "error": str(e)}

    def modify_wrap_reset(self, code_str: str) -> dict:
        local_scope = {}
        clean_code = textwrap.dedent(code_str)
        try:
            exec(clean_code, globals=globals(), locals=local_scope)
            if "reset" in local_scope:
                bound_method = types.MethodType(local_scope["reset"], self)
                self.reset = bound_method
                return {"success": True}
            else:
                return {
                    "success": False,
                    "error": "reset function not found in provided code",
                }
        except Exception as e:
            print(f"Error in modify_reset_fn: {e}")
            return {"success": False, "error": str(e)}


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    wrapper = AgentModifiable(env)
    print(wrapper.env_info()["step_fn"])
    print(inspect.getsource(wrapper.unwrapped.__class__.step))
