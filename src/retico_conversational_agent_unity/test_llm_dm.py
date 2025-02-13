import datetime
import time
import retico_core
import torch
from llama_cpp import Llama
import traceback

import matplotlib.pyplot as plt

from retico_conversational_agent_unity.dialogue_history import DialogueHistory

conv_list = [
    ("Person B", ""),
    (
        "Person A",
        "okay so um yes we do keep uh well we started out keeping a budget about two years ago we have a computer here at the house and i made a Lotus spreadsheet and went through the year using all of our our checkbook to figure out what we spent each time and whether we were over or under for each month",
    ),
    ("Person B", "uh-huh uh-huh"),
    (
        "Person A",
        "and then basically since then what i've done is is keep track of it through the checkbook so that based on whatever we've got coming in check coming in and how much i'm spending each half of the month and then trying to also spend- and because our house payment is once a month that's our our biggest uh expense so i take half of that amount out of my checkbook each with each paycheck even though it's really still there",
    ),
    ("Person B", "uh-huh uh-huh"),
    ("Person A", "so that i can keep a a good balance running total yeah through the month what do y'all do"),
    (
        "Person B",
        "a running total yeah uh we've we've uh taken how much we have you know write down how much we have coming in each month and then uh we've at the beginning of the year we sat down and determined how much we could spend we sat down- made up different accounts like you know we've set a budget for each you know household expenses or food and clothing and entertainment and then our our own fun money and just stuff like that and then we write down each each time we spend something we write down in a book and end of the month we tally it up to see if how close we've you know we we try to stay within a certain budget so",
    ),
    ("Person A", "um-hum is it is it hard to keep track of it or does it work out pretty well"),
    ("Person B", "um it takes some it takes some dedication to do it but it it works out real well so"),
    ("Person A", "um-hum and and you're staying within your budget and keep everything is working pretty good"),
    ("Person B", "uh-huh yeah yeah i stay within- i have to stay within it so i"),
    ("Person A", "yeah i found-"),
    (
        "Person B",
        "you know and then we have that you know if you can't stay if something comes up and you can't stay within it then we have uh you know a budget for you know like we call our slush fund or something and something- unexpected- unexpected comes up then you're not",
    ),
    ("Person A", "um-hum yeah"),
    ("Person B", "you know you don't feel it so strapped"),
    ("Person A", "you don't have to go out and borrow it somewhere and and do that"),
    (
        "Person B",
        "right yeah because we don't you know we don't charge anything that we can't pay off by the end of the month",
    ),
    (
        "Person A",
        "yeah that's a good choice we've been trying we're trying to uh do that this year we've budgeted the money that we used to spend we were spending on a CODA account with TI and then money we were also buying stock with for that year we've taken that this year and said we're gonna pay off all of our credit cards and uh",
    ),
    ("Person B", "uh-huh <b_aside> you got paper under your table <e_aside> uh-huh"),
    (
        "Person A",
        "we have a another loan with the bank and so we hope by the end of this year that by doing that we'll be free and clear",
    ),
    (
        "Person B",
        "uh-huh to be out of debt free yeah the only thing we have it to pay off is our is a automobile loan and our house payment and that's the only thing we ever we try to stay out of debt so",
    ),
    ("Person A", "yeah that's good to be in that kind of shape what are y'all trying to do long term"),
    (
        "Person B",
        "uh-huh oh as long term we just he has- you know his retirement plan and then to CODA and stuff like that that's all we've and you know we just have our life insurance for right now",
    ),
    ("Person A", "uh-huh"),
    ("Person B", "so we don't have any long term you know in stocks or anything like that right now so"),
    (
        "Person A",
        "yeah mostly what we're doing we've worked we've done the uh CODA account with TI where they we put in so much a month and then they or so much a paycheck and then they match it",
    ),
    ("Person B", "yeah that's what we're doing so"),
    (
        "Person A",
        " and so that that has worked out pretty good and then i used to work for TI and i have when i retired from there or left i took the money that i had in mine and put it in an IRA and we had an out",
    ),
    ("Person B", "yeah uh-huh"),
    (
        "Person A",
        "we had an existing IRA so we have both of us have some money in an IRA that we're also trying to figure to put it we're putting it in CDs right now and then we're also looking at it in possibly getting a mutual fund ",
    ),
    ("Person B", "uh-huh yeah whenever we get enough saved we we stick it in a CD for a while and then uh"),
    ("Person A", " um-hum"),
    ("Person B", "you know and then when we if we need it we wait till it it's expired and then so"),
    (
        "Person A",
        "yeah the other thing that we've done that that was really nice to see we had one of the financial companies um  Hancock- oh John Hancock company came out and their agents did a long term analysis based on salary and uh what we were planning- what  what what our uh goals were on a long term budget in terms of retirement kid's college paying off the house buying a different house ",
    ),
    ("Person B", "uh-huh"),
    (
        "Person A",
        "um special thing buying land and building our own house and they did an analysis for us based on what we were putting in and the time frame that we wanted to look at",
    ),
    ("Person B", "uh-huh"),
    (
        "Person A",
        "and then gave us a good idea back you know some good information back on whether or not we were going to achieve those goals and yeah or not or what we needed to do so that we could achieve them and money we could put in at what time",
    ),
    (
        "Person B",
        "or not yeah uh-huh that sounds interesting we've never done anything- we have you know just our our life insurance guy has come out you know and he's set up uh you know determined how much we need to ",
    ),
    ("Person A", "um-hum"),
    ("Person B", "you know we need if something were to happen"),
    ("Person A", "um-hum yeah that"),
    ("Person B", "you know"),
    (
        "Person A",
        "that's the other financial thing i guess that we've done is with our life insurance is since i'm at home now is is figuring out uh what we would need if something happened to my husband or what he would need if something happened to me",
    ),
    (
        "Person B",
        "yeah right yeah you- you know if i would sell the you know if he something would happen to him i wouldn't stay in Texas i would uh",
    ),
    ("Person A", "that's a a big thing to think about"),
    (
        "Person B",
        "sell the house and move back home you know to my home town and and uh i wouldn't stay here in Texas so",
    ),
    ("Person A", "um-hum yeah"),
    ("Person B", "you know i don't know what he would do but"),
    ("Person A", "okay i guess that's most of my um financial plans right now is is there anything you'd like to add"),
    ("Person B", "yeah mine too nope that's about all for mine"),
    ("Person A", "-okay well it's been nice talking to you"),
    ("Person B", "nice talking to you too bye-bye"),
    ("Person A", "bye-bye"),
]


def generate_sentence(model, prompt, reset=True):
    start_tokenize = datetime.datetime.now()
    prompt_tokens = model.tokenize(bytes(prompt, encoding="utf-8"))
    end_tokenize = datetime.datetime.now()
    print("NB tokens = ", len(prompt_tokens))
    tokenize_duration = end_tokenize - start_tokenize
    # pattern = bytes("\n\nChild:", encoding="utf-8")
    pattern = bytes("\n\n", encoding="utf-8")
    pattern_tokens = model.tokenize(pattern, add_bos=False)

    sentence = b""
    sentence_tokens = []
    generate_duration = None
    first_clause_duration = None
    first_clause_completed = False
    try:
        start_generate = datetime.datetime.now()
        for t in model.generate(
            prompt_tokens,
            top_k=40,
            top_p=0.95,
            temp=1.0,
            repeat_penalty=1.1,
            reset=reset,
        ):
            sentence_tokens.append(t)

            # method 2
            word_bytes = model.detokenize([t])
            word = word_bytes.decode(
                "utf-8", errors="ignore"
            )  # special tokens like 243 can raise an error, so we decide to ignore
            sentence += word_bytes

            if not first_clause_completed:

                if word_bytes in punctuation_text:
                    first_clause_completed = True
                    end_first_clause = datetime.datetime.now()
                    first_clause_duration = end_first_clause - start_generate

            if pattern_tokens == sentence_tokens[-len(pattern_tokens) :]:
                break
            if pattern == sentence[-len(pattern) :]:
                break
        end_generate = datetime.datetime.now()
        generate_duration = end_generate - start_generate

        # print(f"prompt = {prompt}")
        # print(
        #     f"DURATIONS = \ntokenize : {tokenize_duration.total_seconds()} \ngenerate : {generate_duration.total_seconds()} \nfirst_clause : {first_clause_duration.total_seconds()}"
        # )
    except Exception:
        print(traceback.format_exc())

    sentence_1 = model.detokenize(sentence_tokens).decode("utf-8")
    # print(f"\n\n sentence : {sentence_1}")

    return (
        sentence_1,
        tokenize_duration.total_seconds(),
        generate_duration.total_seconds() if generate_duration is not None else 0,
        first_clause_duration.total_seconds() if first_clause_duration is not None else 0,
    )


def run_multiple_turns(model, first_turn, nb_turns, dh):
    durations = []
    for i in range(nb_turns):
        prompt = dh.get_prompt()
        sentence, td, gt, ct = generate_sentence(model, prompt, reset=True)
        durations.append([td, gt, ct])
        llm_u = {
            "turn_id": first_turn + i,
            "speaker": "agent",
            "text": sentence,
        }
        dh.append_utterance(llm_u)
        user_u = utterances[first_turn + i * 2 + 1]
        print(f"user_u id={first_turn + i * 2 + 1}, sentence={user_u}")
        dh.append_utterance(user_u)
    return durations


context_size = 8000
nb_turns = 10
log_folder = "logs/run"
terminal_logger, _ = retico_core.log_utils.configurate_logger(log_folder)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"
prompt_format_config = "./configs/prompt_format_config_test.json"

model = Llama(
    model_path=model_path,
    n_ctx=context_size,
    n_gpu_layers=100,
    verbose=True,
)

system_prompt = "This is a spoken dialog scenario between two persons. \
Please provide the next valid response for the following conversation.\
You play the role of Person B. Here is the beginning of the conversation :"


first_10_DH = DialogueHistory(
    prompt_format_config_file=prompt_format_config,
    terminal_logger=terminal_logger,
    initial_system_prompt=system_prompt,
    context_size=context_size,
)
full_DH_exclude_last_nb_turns = DialogueHistory(
    prompt_format_config_file=prompt_format_config,
    terminal_logger=terminal_logger,
    initial_system_prompt=system_prompt,
    context_size=context_size,
)
full_DH = DialogueHistory(
    prompt_format_config_file=prompt_format_config,
    terminal_logger=terminal_logger,
    initial_system_prompt=system_prompt,
    context_size=context_size,
)
longer_DH = DialogueHistory(
    prompt_format_config_file=prompt_format_config,
    terminal_logger=terminal_logger,
    initial_system_prompt=system_prompt,
    context_size=context_size,
)
longer_DH_size = context_size - 1500
utterances = []
for i, turn in enumerate(conv_list):
    role, sentence = turn
    u = {
        "turn_id": i,
        "speaker": "agent" if role == "Person B" else "user",
        "text": sentence,
    }
    longer_DH.append_utterance(u)
    full_DH.append_utterance(u)
    if i < len(conv_list) - (nb_turns * 2):
        print("nb turns = ", len(conv_list) - (nb_turns * 2))
        full_DH_exclude_last_nb_turns.append_utterance(u)
    utterances.append(u)
    if i <= 10:
        first_10_DH.append_utterance(u)

# construct longer_DH
dh_size = len(model.tokenize(bytes(longer_DH.get_prompt(), encoding="utf-8")))
i = 0
while dh_size < longer_DH_size:
    longer_DH.append_utterance(utterances[i])
    i = i + 1 if i < len(utterances) - 2 else 0
    dh_size = len(model.tokenize(bytes(longer_DH.get_prompt(), encoding="utf-8")))

print("\n\nDialogue Histories completed")
print(
    f"size DHs : first_10_DH {len(first_10_DH.get_dialogue_history())},\
    full_DH_exclude_last_nb_turns {len(full_DH_exclude_last_nb_turns.get_dialogue_history())},\
    full_DH {len(full_DH.get_dialogue_history())},\
    longer_DH {len(longer_DH.get_dialogue_history())},\
    nb utterances {len(utterances)}"
)

punctuation_text = [b".", b",", b";", b":", b"!", b"?", b"..."]
punctuation_ids = [b[0] for b in punctuation_text]

durations = []
nb_tokens_in_DH = []
nb_turns_in_DH = []

# first_10_DH
first_turn = 10
dh = first_10_DH
nb_turns_start = len(dh.get_dialogue_history()) - 1
nb_token_start = len(model.tokenize(bytes(dh.get_prompt(), encoding="utf-8")))
durations.append(run_multiple_turns(model, first_turn, nb_turns, dh))
nb_turns_end = len(dh.get_dialogue_history()) - 1
nb_token_end = len(model.tokenize(bytes(dh.get_prompt(), encoding="utf-8")))
nb_turns_in_DH.append([nb_turns_start, nb_turns_end])
nb_tokens_in_DH.append([nb_token_start, nb_token_end])
print(f"n\n###########################################################\n\nprint result : {dh.get_prompt()}")
model.reset()

# full_DH_exclude_last_nb_turns
first_turn = len(utterances) - nb_turns * 2
dh = full_DH_exclude_last_nb_turns
nb_turns_start = len(dh.get_dialogue_history()) - 1
nb_token_start = len(model.tokenize(bytes(dh.get_prompt(), encoding="utf-8")))
durations.append(run_multiple_turns(model, first_turn, nb_turns, dh))
nb_turns_end = len(dh.get_dialogue_history()) - 1
nb_token_end = len(model.tokenize(bytes(dh.get_prompt(), encoding="utf-8")))
nb_turns_in_DH.append([nb_turns_start, nb_turns_end])
nb_tokens_in_DH.append([nb_token_start, nb_token_end])
print(f"\n\n###########################################################\n\nprint result : {dh.get_prompt()}")
model.reset()

# full_DH
first_turn = len(utterances) - nb_turns * 2
dh = full_DH
nb_turns_start = len(dh.get_dialogue_history()) - 1
nb_token_start = len(model.tokenize(bytes(dh.get_prompt(), encoding="utf-8")))
durations.append(run_multiple_turns(model, first_turn, nb_turns, dh))
nb_turns_end = len(dh.get_dialogue_history()) - 1
nb_token_end = len(model.tokenize(bytes(dh.get_prompt(), encoding="utf-8")))
nb_turns_in_DH.append([nb_turns_start, nb_turns_end])
nb_tokens_in_DH.append([nb_token_start, nb_token_end])
print(f"n\n###########################################################\n\nprint result : {dh.get_prompt()}")
model.reset()

# longer_DH
first_turn = len(utterances) - nb_turns * 2
dh = longer_DH
nb_turns_start = len(dh.get_dialogue_history()) - 1
nb_token_start = len(model.tokenize(bytes(dh.get_prompt(), encoding="utf-8")))
durations.append(run_multiple_turns(model, first_turn, nb_turns, dh))
nb_turns_end = len(dh.get_dialogue_history()) - 1
nb_token_end = len(model.tokenize(bytes(dh.get_prompt(), encoding="utf-8")))
nb_turns_in_DH.append([nb_turns_start, nb_turns_end])
nb_tokens_in_DH.append([nb_token_start, nb_token_end])
print(f"n\n###########################################################\n\nprint result : {dh.get_prompt()}")
model.reset()

print(
    f"nb_turns_in_DH :\
    \nDH_first_10 : start: {nb_turns_in_DH[0][0]}, end: {nb_turns_in_DH[0][1]}\
    \nDH_last_10 : start: {nb_turns_in_DH[1][0]}, end: {nb_turns_in_DH[1][1]}\
    \nfull_DH : start: {nb_turns_in_DH[2][0]}, end: {nb_turns_in_DH[2][1]}\
    \nlonger_DH : start: {nb_turns_in_DH[3][0]}, end: {nb_turns_in_DH[3][1]}"
)

print(
    f"nb_tokens_in_DH :\
    \nDH_first_10 : start: {nb_tokens_in_DH[0][0]}, end: {nb_tokens_in_DH[0][1]}\
    \nDH_last_10 : start: {nb_tokens_in_DH[1][0]}, end: {nb_tokens_in_DH[1][1]}\
    \nfull_DH : start: {nb_tokens_in_DH[2][0]}, end: {nb_tokens_in_DH[2][1]}\
    \nlonger_DH : {nb_tokens_in_DH[3][0]}, end: {nb_tokens_in_DH[3][1]}"
)
print(durations)
print(
    f"mean durations :\
    \nDH_first_10 : {sum([d[1] for d in durations[0]])/nb_turns}\
    \nDH_last_10 : {sum([d[1] for d in durations[1]])/nb_turns}\
    \nfull_DH : {sum([d[1] for d in durations[2]])/nb_turns}\
    \nlonger_DH : {sum([d[1] for d in durations[3]])/nb_turns}"
)
print(
    f"mean durations first clause :\
    \nDH_first_10 : {sum([d[2] for d in durations[0]])/nb_turns}\
    \nDH_last_10 : {sum([d[2] for d in durations[1]])/nb_turns}\
    \nfull_DH : {sum([d[2] for d in durations[2]])/nb_turns}\
    \nlonger_DH : {sum([d[2] for d in durations[2]])/nb_turns}"
)

x = list(range(nb_turns))
print(x)
fig, ax = plt.subplots()
ax.plot(x, [d[1] for d in durations[0]], "blue", label="DH_first_10")
ax.plot(x, [d[2] for d in durations[0]], "blue", alpha=0.2, label="DH_first_10 first_clause")
ax.plot(x, [d[1] for d in durations[1]], "brown", label="DH_until_last_10")
ax.plot(x, [d[2] for d in durations[1]], "brown", alpha=0.2, label="DH_until_last_10 first_clause")
ax.plot(x, [d[1] for d in durations[2]], "forestgreen", label="full_DH")
ax.plot(x, [d[2] for d in durations[2]], "forestgreen", alpha=0.2, label="full_DH first_clause")
ax.plot(x, [d[1] for d in durations[3]], "darkviolet", label="longer_DH")
ax.plot(x, [d[2] for d in durations[3]], "darkviolet", alpha=0.2, label="longer_DH first_clause")
# ax.plot(x, [d[1] for d in durations[3]], "forestgreen", label="augmented_DH ")
# ax.plot(x, [d[2] for d in durations[3]], "forestgreen", alpha=0.2, label="augmented_DH first_clause")
ax.set_xlabel("id run")
ax.set_ylabel("duration (seconds)")
ax.legend()
plt.show()
