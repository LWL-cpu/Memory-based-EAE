import openai

# 替换为你的OpenAI API密钥
openai.api_key = 'your-api-key'

def generate_instruction(event_data):
    roles = ', '.join(event_data['role'])
    prompt = f"{event_data['event_type']}: This event has {len(event_data['role'])} arguments ({roles})."

    # 调用ChatGPT API生成每个角色的描述
    roles_descriptions = {}
    for role in event_data['role']:
        role_prompt = f"Please describe the role '{role}' in the context of the event type '{event_data['event_type']}'."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=role_prompt,
            max_tokens=50
        )
        roles_descriptions[role] = response.choices[0].text.strip()

    for role, description in roles_descriptions.items():
        prompt += f" {role}: {description}."

    return prompt

# 示例数据
event_data = {
    "event_type": "Life.Die",
    "trigger": "killed",
    "role": ["Agent", "Victim", "Instrument", "Place"]
}

# 生成指令并存入到input字典中
instruction = generate_instruction(event_data)
event_data['instruction'] = instruction

print(event_data)