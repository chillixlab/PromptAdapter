from prompt_adapter.ai_model_runner import extract_response_text


class _ResponseMessage:
    def __init__(self, content: str):
        self.content = content


class _ResponseChoice:
    def __init__(self, message: _ResponseMessage):
        self.message = message


class _ResponseObject:
    def __init__(self, output_text: str):
        self.output_text = output_text


def test_output_text属性がある場合はその値を返すこと() -> None:
    # GIVEN
    response = _ResponseObject(output_text="これは回答です")

    # WHEN
    text = extract_response_text(response)

    # THEN
    assert text == "これは回答です"


def test_choices形式の辞書レスポンスから本文を抽出できること() -> None:
    # GIVEN
    response = {
        "choices": [
            {
                "message": {
                    "content": "choices形式の回答",
                }
            }
        ]
    }

    # WHEN
    text = extract_response_text(response)

    # THEN
    assert text == "choices形式の回答"


def test_output形式の辞書レスポンスから本文を抽出できること() -> None:
    # GIVEN
    response = {
        "output": [
            {
                "content": [
                    {"text": "1行目"},
                    {"text": "2行目"},
                ]
            }
        ]
    }

    # WHEN
    text = extract_response_text(response)

    # THEN
    assert text == "1行目\n2行目"


def test_choices属性を持つオブジェクトレスポンスから本文を抽出できること() -> None:
    # GIVEN
    response = type(
        "ObjectResponse",
        (),
        {
            "choices": [
                _ResponseChoice(
                    message=_ResponseMessage(content="オブジェクトchoices形式の回答")
                )
            ]
        },
    )()

    # WHEN
    text = extract_response_text(response)

    # THEN
    assert text == "オブジェクトchoices形式の回答"
