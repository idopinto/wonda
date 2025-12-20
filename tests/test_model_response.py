from src.eval.model import InvariantGeneratorModel
from src.utils.program import Program


def test_model_response():
    model = InvariantGeneratorModel(
        client="hf",
        model_cfg={
            "base_model_name": "openai/gpt-oss-20b",
            "eval_finetuned_model": False,
            "is_lora": False,
        },
    )
    program = Program.from_file("tests/data/programs/2048.c")
    response = model.predict(program)
    assert response is not None
    assert response["reasoning"] is not None
    assert response["answer"] is not None
    print(response)


if __name__ == "__main__":
    test_model_response()
