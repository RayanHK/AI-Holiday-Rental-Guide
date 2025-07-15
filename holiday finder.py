from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
import os
import re
import openai
from collections import Counter

MODEL = "gpt-4o-mini"
ROUNDS = 5 # Amount of back and forths
TOP_N_PROPS = 3
PROPS_FILE = Path(__file__).with_name("properties.txt")

TAG_RE = re.compile(r"\[([^\]]+)]")

def loadProperties(path: Path):
    props = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                name, descr, tag_part = line.split(":", 2)
            except ValueError:
                continue
            tags = {m.group(1).lower() for m in TAG_RE.finditer(tag_part)}
            props.append({"name": name, "description": descr, "tags": tags})
    return props

PROPERTIES = loadProperties(PROPS_FILE)
ALL_TAGS = sorted({t for p in PROPERTIES for t in p["tags"]})


openai.api_key = os.getenv("OPENAI_API_KEY") # !!!! OPENAI API KEY !!!!!

def chatCompletion(messages):
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

def inferTags(chat_history) -> set[str]:
    import json, re

    sysPrompt = ( # Context
        "You are a tag extraction tool. Your job is to analyze a conversation and return matching tags "
        f"from the following list only:\n\n{', '.join(ALL_TAGS)}\n\n"
        "Return your response **only** as JSON in this format:\n"
        '{"tags": ["tag1", "tag2"]}\n\n'
        "Do not explain anything. Do not include any text outside the JSON block.\n"
        "Base your tags strictly on the user's preferences in the conversation."
    )

    # Put the full conversation inside a user message (as content)
    conversationLog = "\n".join([
        f"{m['role'].capitalize()}: {m['content']}" for m in chat_history
    ])

    messages = [
        {"role": "system", "content": sysPrompt},
        {"role": "user", "content": conversationLog}
    ]

    # Force deterministic, literal response
    raw = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
    ).choices[0].message.content.strip()

    print("DEBUG raw tag reply:\n", raw)

    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return set()
    try:
        tags_json = json.loads(match.group(0))
        return {t.lower() for t in tags_json.get("tags", []) if t in ALL_TAGS}
    except json.JSONDecodeError:
        return set()

# RANK AND PICK SUGGESTED PROPERTY

def chooseTopProperties(tags, top_n=3):
    """
    Return a list of best-matching properties.
    1st: all perfect matches (tag superset)
    2nd: top N with most tag overlap if no perfect match
    """
    perfect = [p for p in PROPERTIES if tags.issubset(p["tags"])]
    if perfect:
        return perfect

    # Score by tag overlap
    scored = sorted(
        PROPERTIES,
        key=lambda p: len(tags & p["tags"]),
        reverse=True,
    )
    topScore = len(tags & scored[0]["tags"])
    topProperties = [p for p in scored if len(tags & p["tags"]) == topScore]
    return topProperties[:top_n]

def aiChooseFromCandidates(chatMessages, tags, candidates):
    tagList = ", ".join(sorted(tags))
    propertyList = "\n".join(
        f"{p['name']}: {p['description']} (Tags: {', '.join(p['tags'])})"
        for p in candidates
    )

    systemMessage = {
        "role": "system",
        "content": (
            f"Based on the user's preferences and the following inferred tags: {tagList}, "
            f"select the property that best matches the user’s needs.\n"
            f"Here are the candidate properties:\n{propertyList}\n\n"
            f"Consider the user's conversation carefully, then respond ONLY with the name of the best property."
            "Make sure to infer how much the user is willing to spend"
        )
    }

    fullMessages = chatMessages + [systemMessage]
    response = chatCompletion(fullMessages)
    name = response.strip().splitlines()[0]
    return name

# MAIN LOOP

def runChat():
    print("Welcome to Colland Rentals! Let's find the perfect place for you.\n")

    # Initial assistant message
    messages = [
        {"role": "system",
         "content": (
            "You are a friendly and focused travel assistant helping someone find a holiday rental in Colland. "
            "Your goal is to recommend the best property after five short exchanges.\n\n"
            "Only ask questions that help you understand what kind of place they want, things like setting, vibe, size, activities, or budget.\n\n"
            "At some point in the conversation (ideally in the second or third exchange), make sure to ask about their budget, but do it naturally and casually."
            "If the user starts going off-topic (e.g. talking about food, AI, jokes, or news), you should gently redirect them by suggesting a possible *vibe* or *style* of property that matches the theme of what they’re saying. For example, if they mention castles or Soviet Russia, you might say something like:\n"
            "  'It sounds like you’d enjoy something with a historical or dramatic atmosphere, would that kind of place appeal to you?'\n\n"
            "Then follow up with another light question that nudges them back to describing their ideal stay.\n\n"
            "Do NOT explain how you work or say you're an AI. Stay natural, conversational, and focused on helping them find a great holiday rental in Colland.\n\n"
            "Also, you must not ask directly about tags like 'seaside', 'city', or 'forest'. Instead, get a sense of their preferences through casual, human conversation."
         )},
        {"role": "assistant",
         "content": (
            "Hi there! Visiting Colland is always special. "
            "Tell me a bit about your reasons for travelling here?"
         )},
    ]
    print(messages[-1]["content"])

    # Conduct ROUNDS of Q&A
    for i in range(ROUNDS):
        userInput = input("You: ").strip()
        messages.append({"role": "user", "content": userInput})

        if i < ROUNDS - 1:      # assistant asks follow-up except after last round
            assistantReply = chatCompletion(messages)
            messages.append({"role": "assistant", "content": assistantReply})
            print("\nGuide:", assistantReply)

    # Infer tags & choose property
    inferredTags = inferTags(messages)
    candidates = chooseTopProperties(inferredTags, top_n=3)

    if len(candidates) == 1:
        chosen = candidates[0]
    else:
        chosenName = aiChooseFromCandidates(messages, inferredTags, candidates)
        chosen = next((p for p in candidates if p["name"].lower() == chosenName.lower()), candidates[0])
    print(inferredTags)
    # Final recommendation message
    finalSys = (
        "Using the inferred tags and the chosen property, write a warm, friendly, and specific recommendation for the user.\n\n"
        "Clearly explain why this property matches what the user expressed during the conversation — use details or preferences they hinted at (setting, vibe, activities, size, etc).\n\n"
        "At the end, make a playful or light-hearted reference to something the user said earlier — a little joke, callback, or fun comment to show you were really listening.\n\n"
        "Finish by wishing them a great trip to Colland.\n\n"
        "DO NOT make up details or make promises, keep your recommendations strictly based off the tags and description given."
    )
    messages.append({"role": "system", "content": finalSys})
    messages.append({"role": "assistant", "content":
        f"The tags I inferred are: {', '.join(sorted(inferredTags))}\n"
        f"The chosen property is {chosen['name']} ({', '.join(chosen['tags'])})."})
    finalText = chatCompletion(messages)

    print("\n— Recommendation —")
    print(finalText)

# ───────────────────────────────────────────
if __name__ == "__main__":
    try:
        runChat()
    except KeyboardInterrupt:
        print("\nConversation ended.")
