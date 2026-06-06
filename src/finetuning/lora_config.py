from unsloth import FastLanguageModel

def apple_lora(model):

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        lora_alpha = 32,
        lora_dropout = 0.2,
        bias = 'none',
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing = "unsloth",
    )

    return model