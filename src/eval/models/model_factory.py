import weave
from omegaconf import DictConfig

from src.eval.models.base_model import InvariantGeneratorModelInterface
from src.eval.models.open_router_model import InvariantGeneratorOpenRouterModel
from src.eval.models.oss_model import InvariantGeneratorOssModel
from src.eval.models.qwen_model import InvariantGeneratorQwenModel
from src.eval.models.together_model import InvariantGeneratorTogetherModel
from src.eval.models.vllm_model import InvariantGeneratorVllmModel


class ModelFactory:
    @staticmethod
    def create(cfg: DictConfig, system_prompt: weave.StringPrompt, user_prompt_template: weave.StringPrompt) -> InvariantGeneratorModelInterface:
        """Create a model instance based on configuration."""
        # Get model name from either id or base_model_name field
        if cfg.client == "together":
            return InvariantGeneratorTogetherModel(
                model_cfg=cfg,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
            )
        if cfg.client == "openrouter":
            return InvariantGeneratorOpenRouterModel(
                model_cfg=cfg,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
            )
        elif cfg.client == "vllm":
            return InvariantGeneratorVllmModel(
                model_cfg=cfg,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
            )
        elif cfg.client == "hf":
            if "qwen" in cfg.base_model.id.lower():
                return InvariantGeneratorQwenModel(
                    model_cfg=cfg,
                    system_prompt=system_prompt,
                    user_prompt_template=user_prompt_template,
                    sampling_params=cfg.base_model.sampling_params,
                    enable_thinking=cfg.base_model.extra_body.enable_thinking,
                    eval_ft_model=cfg.eval_ft_model,
                )
            elif "gpt-oss" in cfg.base_model.id.lower():
                return InvariantGeneratorOssModel(
                    model_cfg=cfg,
                    system_prompt=system_prompt,
                    user_prompt_template=user_prompt_template,
                    sampling_params=cfg.base_model.sampling_params,
                    reasoning_effort=cfg.base_model.reasoning_effort,
                    eval_ft_model=cfg.eval_ft_model,
                )
            else:
                raise ValueError(f"Invalid model: {cfg.base_model.id}")
        else:
            raise ValueError(f"Invalid client: {cfg.client}")


# if __name__ == "__main__":
#     # print("Hello, World!")
#     from openai import OpenAI
#     import os
#     from dotenv import load_dotenv
#     load_dotenv()
#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#     response = client.responses.create(
#     model="gpt-4.1",
#     input="Tell me a three sentence bedtime story about a unicorn."
#     )

#     print(response)
