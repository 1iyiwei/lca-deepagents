from collections import deque

# my own utils

# get the provider (e.g. "anthropic", "google_genai", "openai") of a model
# created via init_chat_model(), regardless of the model string used to init it
def get_provider(model):
    return model._get_ls_params()["ls_provider"]


# breadth-first search for the shallowest "text" key in a nested dict/list structure
def _find_shallowest_text(content, text_key="text"):
    queue = deque([content])
    while queue:
        current = queue.popleft()
        if isinstance(current, dict):
            if text_key in current:
                return current[text_key]
            queue.extend(current.values())
        elif isinstance(current, (list, tuple)):
            queue.extend(current)
    raise ValueError(f"no {text_key} key found")


def get_content(model, message):
    text_key = "text"
    if get_provider(model) == "google_genai":
        try:
            return message.content[0][text_key]
        except TypeError:
            try:
                return _find_shallowest_text(message.content, text_key)
            except Exception:
                return message.content
    else:
        return message.content

# print the content of a message only without other extra meta data based on the model's provider
def print_content(model, message):
    print(get_content(model, message))

# invoke an agent, gracefully reporting errors instead of crashing
def invoke_agent(agent, input):
    try:
        return agent.invoke(input)
    except Exception as e:
        print(f"Invocation failed: {e}")
        return None