import json

valid_examples = 0
with open("data/raw/cs_tutor_seed.jsonl", "r", encoding="utf-8") as reader:
    for line_number, line in enumerate(reader, start=1):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            print(f"Line {line_number}: invalid JSON")
            continue
        if "messages" not in data:
            print(f"Line {line_number}: missing 'messages' field")
            continue
        if not isinstance(data["messages"], list):
            print(f"Line {line_number}: 'messages' field is not a list")
            continue
        if len(data["messages"]) < 2:
            print(f"Line {line_number}: 'messages' field has fewer than 2 messages")
            continue
        is_valid = True
        for message in data["messages"]:
            if not isinstance(message, dict):
                print(f"Line {line_number}: message is not a dictionary")
                is_valid = False
                continue
            if "role" not in message:
                print(f"Line {line_number}: message missing 'role' field")
                is_valid = False
                continue
            if "content" not in message:
                print(f"Line {line_number}: message missing 'content' field")
                is_valid = False
                continue
            if not isinstance(message["role"], str):
                print(f"Line {line_number}: 'role' field is not a string")
                is_valid = False
                continue
            if not isinstance(message["content"], str):
                print(f"Line {line_number}: 'content' field is not a string")
                is_valid = False
                continue
            if message["role"] not in ("user", "assistant"):
                print(f"Line {line_number}: invalid role '{message['role']}'")
                is_valid = False
                continue
            if not message["content"].strip():
                print(f"Line {line_number}: message content is empty")
                is_valid = False
                continue
        roles = {message["role"] for message in data["messages"] if isinstance(message, dict) and "role" in message}
        if "user" not in roles or "assistant" not in roles:
            print(f"Line {line_number}: must contain both user and assistant messages")
            is_valid = False
        if is_valid:
            valid_examples += 1
print(f"Validated {valid_examples} examples successfully.")