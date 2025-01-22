"""
Dialogue History
================

The dialogue history can be used to store every user and agent previous
turns during a dialogue. You can add data using update its data using
the append_utterance function, update the current turn stored using
prepare_dialogue_history and get the updates prompt using get_prompt.

The DialogueHistory is using a template config file, that you can change
to configure the prefixes, suffixes, roles, for the user, agent, system
prompt and the prompt itself. It is useful because every LLm has a
different prefered template for its prompts.

Example of a prompt with the following config :
{
"user": {
"role": "Child",
"role_sep": ":",
"pre": "",
"suf": "\n\n"
},
"agent": {
"role": "Teacher",
"role_sep": ":",
"pre": "",
"suf": "\n\n"
},
"system_prompt": {
"pre": "<<SYS>>\n",
"suf": "<</SYS>>\n\n"
},
"prompt": {
"pre": "[INST] ",
"suf": "[/INST]"
}
}

prompt = "[INST] <<SYS>>
This is a spoken dialog scenario between a teacher and a 8 years old 
child student. The teacher is teaching mathematics to the child student.
As the student is a child, the teacher needs to stay gentle all the
time. Please provide the next valid response for the following
conversation. You play the role of a teacher. Here is the beginning of
the conversation : 
<</SYS>>

Child : Hello !

Teacher : Hi! How are you today ?

Child : I am fine, and I can't wait to learn mathematics!

[/INST]
Teacher :"
"""

import json


class DialogueHistory:
    """The dialogue history is where all the sentences from the previvous agent
    and user turns will be stored.

    The LLM, and/or DM will retrieve the history to build the prompt and
    use it to generate the next agent turn.
    """

    def __init__(
        self,
        prompt_format_config_file,
        terminal_logger,
        file_logger=None,
        initial_system_prompt="",
        context_size=2000,
    ):
        """Initializes the DialogueHistory.

        Args:
            prompt_format_config_file (str): path to prompt template
                config file.
            terminal_logger (TerminalLogger): The logger used to print
                events in console.
            file_logger (FileLogger, optional): The logger used to store
                events in a log file.. Defaults to None.
            initial_system_prompt (str, optional): The initial system
                prompt containing the dialogue scenario and/or
                instructions. Defaults to "".
            context_size (int, optional): Max number of tokens that the
                total prompt can contain (LLM context size). Defaults to
                2000. Defaults to 2000.
        """
        self.terminal_logger = terminal_logger
        self.file_logger = file_logger
        with open(prompt_format_config_file, "r", encoding="utf-8") as config:
            self.prompt_format_config = json.load(config)
        self.initial_system_prompt = initial_system_prompt
        self.current_system_prompt = initial_system_prompt
        self.dialogue_history = [
            {
                "turn_id": -1,
                "speaker": "system_prompt",
                "text": initial_system_prompt,
            }
        ]
        self.cpt_0 = 1
        self.context_size = context_size

    # Formatters

    def format_role(self, config_id):
        """Function that format a sentence by adding the role and role
        separation.

        Args:
            config_id (str): the id to find the corresponding prefix,
                suffix, etc in the config.

        Returns:
            str: the formatted sentence.
        """
        if "role" in self.prompt_format_config[config_id]:
            return (
                self.prompt_format_config[config_id]["role"]
                + " "
                + self.prompt_format_config[config_id]["role_sep"]
                + " "
            )
        else:
            return ""

    def format(self, config_id, text):
        """Basic function to format a text with regards to the
        prompt_format_config. Format meaning to add prefix, sufix, role, etc to
        the text (for agent or user sentence, system prompt, etc).

        Args:
            config_id (str): the id to find the corresponding prefix,
                suffix, etc in the config.
            text (str): the text to format with the
                prompt_format_config.

        Returns:
            str: the formatted text.
        """
        return (
            self.prompt_format_config[config_id]["pre"]
            + self.format_role(config_id)
            + text
            + self.prompt_format_config[config_id]["suf"]
        )

    def format_sentence(self, utterance):
        """Function that formats utterance, to whether an agent or a user
        sentence.

        Args:
            utterance (dict[str]): a dictionary describing the utterance
                to format (speaker, and text).

        Returns:
            str: the formatted sentence.
        """
        return self.format(config_id=utterance["speaker"], text=utterance["text"])

    # Setters

    def append_utterance(self, utterance):
        """Add the utterance to the dialogue history.

        Args:
            utterance (dict): a dict containing the speaker and the
                turn's transcription (text of the sentences).
        """
        assert set(("turn_id", "speaker", "text")) <= set(utterance)
        # insure that turn_id is not None, and increment turn_id for system that do not have a turn id cpt (like DM).
        utterance["turn_id"] = (
            len(self.dialogue_history) if utterance["turn_id"] else utterance["turn_id"]
        )
        self.dialogue_history.append(utterance)
        c = self.prompt_format_config
        s = utterance["speaker"]
        print(f"\n{c[s]['role']} {c[s]['role_sep']} {utterance['text']}")

    def reset_system_prompt(self):
        """Set the system prompt to initial_system_prompt, which is the prompt
        given at the DialogueHistory initialization."""
        self.change_system_prompt(self.initial_system_prompt)

    def change_system_prompt(self, system_prompt):
        """Function that changes the DialogueHistory current system prompt. The
        system prompt contains the LLM instruction and the scenario of the
        interaction.

        Args:
            system_prompt (str): the new system_prompt.

        Returns:
            str: the previous system_prompt.
        """
        previous_system_prompt = self.current_system_prompt
        self.current_system_prompt = system_prompt
        self.dialogue_history[0]["text"] = system_prompt
        return previous_system_prompt

    def prepare_dialogue_history(self, fun_tokenize):
        """Calculate if the current dialogue history is bigger than the LLM's
        context size (in nb of token). If the dialogue history contains too
        many tokens, remove the older dialogue turns until its size is smaller
        than the context size. The self.cpt_0 class argument is used to store
        the id of the older turn of last prepare_dialogue_history call (to
        start back the while loop at this id).

        Args:
            fun_tokenize (Callable[]): the tokenize function given by
                the LLM, so that the DialogueHistory can calculate the
                right dialogue_history size.

        Returns:
            (text, int): the prompt to give to the LLM (containing the
                formatted system prompt, and a maximum of formatted
                previous sentences), and it's size in nb of token.
        """

        prompt = self.get_prompt(self.cpt_0)
        prompt_tokens = fun_tokenize(bytes(prompt, "utf-8"))
        nb_tokens = len(prompt_tokens)
        while nb_tokens > self.context_size:
            self.cpt_0 += 1
            prompt = self.get_prompt(self.cpt_0)
            prompt_tokens = fun_tokenize(bytes(prompt, "utf-8"))
            nb_tokens = len(prompt_tokens)
        return prompt, prompt_tokens

    def interruption_alignment_new_agent_sentence(
        self, utterance, punctuation_ids, interrupted_speaker_iu
    ):
        """After an interruption, this function will align the sentence stored
        in dialogue history with the last word spoken by the agent. With the
        informations stored in interrupted_speaker_iu, this function will
        shorten the utterance to be aligned with the last words spoken by the
        agent.

        Args:
            utterance (dict[str]): the utterance generated by the LLM,
                that has been interrupted by the user and needs to be
                aligned.
            punctuation_ids (list[int]): the id of the punctuation
                marks, calculated by the LLM at initialization.
            interrupted_speaker_iu (IncrementalUnit): the
                SpeakerModule's IncrementalUnit, used to align the agent
                utterance.
        """
        new_agent_sentence = utterance["text"].encode("utf-8")

        # split the sentence into clauses
        sentence_clauses = []
        old_i = 0
        for i, c in enumerate(new_agent_sentence):
            if c in punctuation_ids or i == len(new_agent_sentence) - 1:
                sentence_clauses.append(new_agent_sentence[old_i : i + 1])
                old_i = i + 1

        # remove all clauses after clause_id (the interrupted clause)
        sentence_clauses = sentence_clauses[: interrupted_speaker_iu.clause_id + 1]

        # Shorten the last agent utterance until the last char outputted by the speakermodule before the interruption
        sentence_clauses[-1] = sentence_clauses[-1][
            : interrupted_speaker_iu.char_id + 1
        ]

        # Merge the clauses back together
        new_agent_sentence = b"".join(sentence_clauses)

        # decode
        new_agent_sentence = new_agent_sentence.decode("utf-8")

        self.terminal_logger.info(
            "DH interruption alignement func",
            utterance=utterance,
            new_agent_sentence=new_agent_sentence,
            debug=True,
        )

        # store the new sentence in the dialogue history
        utterance["text"] = new_agent_sentence
        self.append_utterance(utterance)

        print("INTERRUPTED AGENT SENTENCE : ", new_agent_sentence)

    # Getters

    def get_dialogue_history(self):
        """Get DialogueHistory's dictionary containing the system prompt and
        all previous turns.

        Returns:
            dict: DialogueHistory's dictionary.
        """
        return self.dialogue_history

    def get_prompt(self, start=1, end=None, system_prompt=None):
        """Get the formatted prompt containing all turns between start and end.

        Args:
            start (int, optional): start id of the oldest turn to take.
                Defaults to 1.
            end (int, optional): end id of the latest turn to take.
                Defaults to None.

        Returns:
            str: the corresponding formatted prompt.
        """
        if end is None:
            end = len(self.dialogue_history)
        assert start > 0
        assert end >= start
        if system_prompt is not None:
            prompt = self.format("system_prompt", system_prompt)
        else:
            prompt = self.format_sentence(self.dialogue_history[0])
        for utterance in self.dialogue_history[start:end]:
            prompt += self.format_sentence(utterance)
        prompt = self.format("prompt", prompt)

        # put additional "/n/nTeacher :" at the end of the prompt, so that it is not the LLM that generates the role
        prompt += (
            "\n\n"
            + self.prompt_format_config["agent"]["pre"]
            + self.format_role("agent")
        )
        return prompt

    def get_stop_patterns(self):
        """Get stop patterns for both user and agent.

        Returns:
            tuple[bytes], tuple[bytes]: user and agent stop patterns.
        """
        c = self.prompt_format_config
        user_stop_pat = (
            bytes(c["user"]["role"] + " " + c["user"]["role_sep"], encoding="utf-8"),
            bytes(c["user"]["role"] + "" + c["user"]["role_sep"], encoding="utf-8"),
        )
        agent_stop_pat = (
            bytes(c["agent"]["role"] + " " + c["agent"]["role_sep"], encoding="utf-8"),
            bytes(c["agent"]["role"] + "" + c["agent"]["role_sep"], encoding="utf-8"),
        )
        return (user_stop_pat, agent_stop_pat)
