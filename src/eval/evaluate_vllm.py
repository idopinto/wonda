from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    load_dotenv()
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, top_k=0.0, max_new_tokens=16384)
    llm = LLM(model="openai/gpt-oss-20b")

    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    messages_list = [
        [{"role": "user", "content": prompt}]
        for prompt in prompts
    ]
    texts = tokenizer.apply_chat_template(
        messages_list,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Generate outputs
    outputs = llm.generate(texts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # Using chat interface.
    outputs = llm.chat(messages_list, sampling_params)
    for idx, output in enumerate(outputs):
        prompt = prompts[idx]
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    main()
