import weave
from omegaconf import DictConfig

from wonda.eval.models.open_router_model import InvariantGeneratorOpenRouterModel
from wonda.eval.models.oss_model import InvariantGeneratorOssModel
from wonda.eval.models.qwen_model import InvariantGeneratorQwenModel
from wonda.eval.models.together_model import InvariantGeneratorTogetherModel
from wonda.eval.models.vllm_model import InvariantGeneratorVllmModel


class ModelFactory:
    @staticmethod
    def create(cfg: DictConfig, system_prompt: weave.StringPrompt, user_prompt_template: weave.StringPrompt) -> weave.Model:
        """Create a model instance based on configuration."""
        # Get model name from either id or base_model_name field
        if cfg.client == "together":
            return InvariantGeneratorTogetherModel(
                model_cfg=cfg,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
            )
        elif cfg.client == "openrouter":
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
