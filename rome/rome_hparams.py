from dataclasses import dataclass, field
from typing import List

from util.hparams import HyperParams


@dataclass
class ROMEHyperParams(HyperParams):
    # Options to enable fixes to key computations
    enable_prompt_keys: bool = field(default=False)
    enable_random_prefix_keys: bool = field(default=True)
    original_implementation: bool = field(default=True)
    
    # Method
    layers: List[int] = field(default_factory=list)
    fact_token: str = field(default="subject_first")
    v_num_grad_steps: int = field(default=20)
    v_lr: float = field(default=5e-1)
    v_loss_layer: int = field(default=-1)
    v_weight_decay: float = field(default=0.5)
    clamp_norm_factor: float = field(default=4.0)
    kl_factor: float = field(default=0.0625)
    mom2_adjustment: bool = field(default=True)
    context_template_length_params: List[List[int]] = field(default_factory=list)

    # Module templates
    rewrite_module_tmp: str = field(default="model.layers.{}.mlp.down_proj")
    layer_module_tmp: str = field(default="model.layers.{}")
    mlp_module_tmp: str = field(default="model.layers.{}.mlp")
    attn_module_tmp: str = field(default="model.layers.{}.self_atten.o_proj"),
    ln_f_module: str = field(default="model.norm")
    lm_head_module: str = field(default="lm_head")

    # Statistics
    mom2_dataset: str = field(default="wikipedia")
    mom2_n_samples: int = field(default=20)
    mom2_dtype: str = field(default="float32")
