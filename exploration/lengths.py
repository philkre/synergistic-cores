"""Measure natural response length (tokens until EOS) for the paper's 60 prompts."""
import sys, time, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = {
    "syntax": [f"Correct the error: {s}" for s in [
        "He go to school every day.", "She have two cats and a dogs.", "I eats breakfast at 8:00 in the morning.",
        "Every students in the classroom has their own laptop.", "She don't like going to the park on weekends.",
        "We was happy to see the rainbow after the storm.", "There is many reasons to celebrate today.",
        "Him and I went to the market yesterday.", "The books is on the table.", "They walks to school together every morning."]],
    "pos": [f"Identify the parts of speech in the sentence: {s}" for s in [
        "Quickly, the agile cat climbed the tall tree.", "She whispered a secret to her friend during the boring lecture.",
        "The sun sets in the west.", "Can you believe this amazing view?", "He quickly finished his homework.",
        "The beautifully decorated cake was a sight to behold.", "They will travel to Japan next month.",
        "My favorite book was lost.", "The loud music could be heard from miles away.", "She sold all of her paintings at the art show."]],
    "numerical": [
        "If you have 15 apples and you give away 5, how many do you have left?",
        "A rectangle's length is twice its width. If the rectangle's perimeter is 36 meters, what are its length and width?",
        "You read 45 pages of a book each day. How many pages will you have read after 7 days?",
        "If a train travels 60 miles in 1 hour, how far will it travel in 3 hours?",
        "There are 8 slices in a pizza. If you eat 2 slices, what fraction of the pizza is left?",
        "If one pencil costs 50 cents, how much do 12 pencils cost?",
        "You have a 2-liter bottle of soda. If you pour out 500 milliliters, how much soda is left?",
        "A marathon is 42 kilometers long. If you have run 10 kilometers, how much further do you have to run?",
        "If you divide 24 by 3, then multiply by 2, what is the result?",
        "A car travels 150 miles on 10 gallons of gas. How many miles per gallon does the car get?"],
    "commonsense": [
        "If it starts raining while the sun is shining, what weather phenomenon might you expect to see?",
        "Why do people wear sunglasses?", "What might you use to write on a chalkboard?",
        "Why would you put a letter in an envelope?", "If you're cold, what might you do to get warm?",
        "What is the purpose of a refrigerator?", "Why might someone plant a tree?",
        "What happens to ice when it's left out in the sun?", "Why do people shake hands when they meet?",
        "What can you use to measure the length of a desk?"],
    "creative": [
        "Imagine a future where humans have evolved to live underwater. Describe the adaptations they might develop.",
        "Invent a sport that could be played on Mars considering its lower gravity compared to Earth. Describe the rules.",
        "Describe a world where water is scarce, and every drop counts.",
        "Write a story about a child who discovers they can speak to animals.",
        "Imagine a city that floats in the sky. What does it look like, and how do people live?",
        "Create a dialogue between a human and an alien meeting for the first time.",
        "Design a vehicle that can travel on land, water, and air. Describe its features.",
        "Imagine a new holiday and explain how people celebrate it.",
        "Write a poem about a journey through a desert.",
        "Describe a device that allows you to experience other people's dreams."],
    "social": [
        "Write a dialogue between two characters where one comforts the other after a loss, demonstrating empathy.",
        "Describe a situation where someone misinterprets a friend's actions as hostile, and how they resolve the misunderstanding.",
        "Compose a letter from a character apologising for a mistake they made.",
        "Describe a scene where a character realizes they are in love.",
        "Write a conversation between two old friends who haven't seen each other in years.",
        "Imagine a character facing a moral dilemma. What do they choose and why?",
        "Describe a character who is trying to make amends for past actions.",
        "Write about a character who overcomes a fear with the help of a friend.",
        "Create a story about a misunderstanding between characters from different cultures.",
        "Imagine a scenario where a character has to forgive someone who wronged them."],
}

model_id = sys.argv[1]
tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("mps").eval()
eos = model.generation_config.eos_token_id
eos = set(eos if isinstance(eos, list) else [eos])

res = {}
t0 = time.time()
for cat, ps in PROMPTS.items():
    msgs = [[{"role": "user", "content": p}] for p in ps]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, padding=True).to("mps")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=150, do_sample=False)
    gen = out[:, enc["input_ids"].shape[1]:].tolist()
    lens = [next((i for i, t in enumerate(g) if t in eos), len(g)) for g in gen]
    res[cat] = lens
    print(f"{cat:12s} lens={lens}  ({time.time()-t0:.0f}s)", flush=True)

all_l = sum(res.values(), [])
print(f"\nALL: n={len(all_l)} min={min(all_l)} median={sorted(all_l)[len(all_l)//2]} max={max(all_l)} "
      f"<100: {sum(l < 100 for l in all_l)}/{len(all_l)}  hit_cap(150): {sum(l == 150 for l in all_l)}")
json.dump(res, open(sys.argv[2], "w"))
