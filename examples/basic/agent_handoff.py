from swarm import Swarm, Agent

client = Swarm()

english_agent = Agent(
    name="English Agent",
    instructions="You only speak English.",
)

spanish_agent = Agent(
    name="Spanish Agent",
    instructions="You only speak Spanish.",
)

hindi_agent = Agent(
    name="Hindi Agent",
    instructions="You only speak Hindi.",
)


def transfer_to_spanish_agent():
    """Transfer spanish speaking users immediately."""
    return spanish_agent

def transfer_to_hindi_agent():
    """Transfer hindi speaking users immediately."""
    return hindi_agent


english_agent.functions.append(transfer_to_spanish_agent)
hindi_agent.functions.append(transfer_to_hindi_agent)

messages = [{"role": "user", "content": "Hola. ¿Como estás?"}]
response = client.run(agent=english_agent, messages=messages)

print(response.messages[-1]["content"])
