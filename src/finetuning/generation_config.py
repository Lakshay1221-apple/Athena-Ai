from transformers import GenerationConfig 

def get_generation_config():

    return GenerationConfig(
        max_new_tokens = 512,
        temperature = 0.7,
        top_p = 0.9,
        top_k = 50,
        do_sample = True,
        repetition_penalty = 1.2,
    )