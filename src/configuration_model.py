from transformers import PretrainedConfig

class ImplicitModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model='gpt2',
        temperature_init_value=1.0,
        temperature_learnable=False,
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = base_model
        self.temperature_init_value = temperature_init_value
        self.temperature_learnable = temperature_learnable
        super().__init__(**kwargs)

