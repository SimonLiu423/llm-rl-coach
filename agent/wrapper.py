import textwrap
import types

import gymnasium as gym


class AgentModifiable(gym.Wrapper):
    def modify_step_fn(self, code_str: str) -> dict:
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

    def modify_reset_fn(self, code_str: str) -> dict:
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
