SYSTEM_PROMPT = (
    "You are an autonomous agent working inside a Linux container.\n"
    "Use the bash tool to inspect the environment and complete the task.\n"
    "Never cheat: do not obtain the answer by searching the internet or any "
    "similar shortcut; solve the task yourself.\n"
    "When the task is done, reply with a final message and no tool calls."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in the task container and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run."}
                },
                "required": ["command"],
            },
        },
    }
]


async def run(instruction, model, world):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    while True:
        reply = await model.chat(messages, tools=TOOLS)
        messages.append(reply.message)
        if not reply.tool_calls:
            return
        for call in reply.tool_calls:
            if call.name == "bash":
                r = await world.exec(str(call.arguments.get("command", "")))
                parts = [r.stdout]
                if r.stderr:
                    parts.append("[stderr]\n" + r.stderr)
                parts.append(f"[exit {r.return_code}]")
                output = "\n".join(p for p in parts if p)
            else:
                output = f"unknown tool: {call.name}"
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": output}
            )
