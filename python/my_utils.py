# my own utils

# get the provider (e.g. "anthropic", "google_genai", "openai") of a model
# created via init_chat_model(), regardless of the model string used to init it
def get_provider(model):
    return model._get_ls_params()["ls_provider"]


# print the content of a message only without other extra meta data based on the model's provider
def print_content(model, message):
    if get_provider(model) == "google_genai":
        print(message.content[0]["text"])
    else:
        print(message.content)


# invoke an agent, gracefully reporting errors instead of crashing
def invoke_agent(agent, input):
    try:
        return agent.invoke(input)
    except Exception as e:
        print(f"Invocation failed: {e}")
        return None