from typing import Any


def extract_response_text(response: Any) -> str:
    """LiteLLMレスポンスから回答本文を抽出する。

    Parameters
    ----------
    response : Any
        LiteLLMから返却されたレスポンス。

    Returns
    -------
    str
        抽出した回答本文。抽出できない場合は文字列化した値。
    """
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip() != "":
        return output_text

    if isinstance(response, dict):
        choice_text = _extract_from_choices(response.get("choices"))
        if choice_text:
            return choice_text

        output_text_from_dict = _extract_from_output(response.get("output"))
        if output_text_from_dict:
            return output_text_from_dict

        content_text = response.get("content")
        if isinstance(content_text, str) and content_text.strip() != "":
            return content_text

    choices = getattr(response, "choices", None)
    choice_text = _extract_from_choices(choices)
    if choice_text:
        return choice_text

    output = getattr(response, "output", None)
    output_text_from_object = _extract_from_output(output)
    if output_text_from_object:
        return output_text_from_object

    return str(response)


def _extract_from_choices(choices: Any) -> str:
    """choices形式のレスポンスから回答本文を抽出する。

    Parameters
    ----------
    choices : Any
        レスポンスのchoices相当データ。

    Returns
    -------
    str
        抽出した回答本文。抽出できない場合は空文字。
    """
    if not isinstance(choices, list) or not choices:
        return ""

    first_choice = choices[0]
    if isinstance(first_choice, dict):
        message = first_choice.get("message", {})
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip() != "":
                return content
        text = first_choice.get("text")
        if isinstance(text, str) and text.strip() != "":
            return text

    message = getattr(first_choice, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip() != "":
            return content

    text = getattr(first_choice, "text", None)
    if isinstance(text, str) and text.strip() != "":
        return text

    return ""


def _extract_from_output(output: Any) -> str:
    """responses形式のoutput配列から回答本文を抽出する。

    Parameters
    ----------
    output : Any
        レスポンスのoutput相当データ。

    Returns
    -------
    str
        抽出した回答本文。抽出できない場合は空文字。
    """
    if not isinstance(output, list):
        return ""

    text_chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for content_item in content:
            if not isinstance(content_item, dict):
                continue
            text = content_item.get("text")
            if isinstance(text, str) and text.strip() != "":
                text_chunks.append(text)

    return "\n".join(text_chunks)
