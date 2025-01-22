"""
TTS Module
==========

A retico module that provides Text-To-Speech (TTS), aligns its inputs
and ouputs (text and audio), and handles user interruption.

When receiving COMMIT TurnTextIUs, synthesizes audio
(TextAlignedAudioIU) corresponding to all IUs contained in
UpdateMessage. This module also aligns the inputed words with the
outputted audio, providing the outputted TextAlignedAudioIU with the
information of the word it corresponds to (contained in the
grounded_word parameter), and its place in the agent's current sentence.
The module stops synthesizing if it receives the information that the
user started talking (user barge-in/interruption of agent turn). The
interruption information is recognized by an VADTurnAudioIU with a
parameter vad_state="interruption".

This modules uses the deep learning approach implemented with coqui-ai's
TTS library : https://github.com/coqui-ai/TTS

Inputs : TurnTextIU, VADTurnAudioIU

Outputs : TextAlignedAudioIU
"""

import datetime
import random
import threading
import time
import numpy as np
from TTS.api import TTS

import retico_core
from retico_core.log_utils import log_exception

from retico_conversational_agent.utils import device_definition
from retico_conversational_agent.additional_IUs import (
    BackchannelIU,
    TurnTextIU,
    VADTurnAudioIU,
    TextAlignedAudioIU,
    DMIU,
)


class TtsDmModule(retico_core.AbstractModule):
    """A retico module that provides Text-To-Speech (TTS), aligns its inputs
    and ouputs (text and audio), and handles user interruption.

    When receiving COMMIT TurnTextIUs, synthesizes audio
    (TextAlignedAudioIU) corresponding to all IUs contained in
    UpdateMessage. This module also aligns the inputed words with the
    outputted audio, providing the outputted TextAlignedAudioIU with the
    information of the word it corresponds to (contained in the
    grounded_word parameter), and its place in the agent's current
    sentence. The module stops synthesizing if it receives the
    information that the user started talking (user barge-
    in/interruption of agent turn). The interruption information is
    recognized by an VADTurnAudioIU with a parameter
    vad_state="interruption".

    This modules uses the deep learning approach implemented with coqui-
    ai's TTS library : https://github.com/coqui-ai/TTS

    Inputs : TurnTextIU, VADTurnAudioIU

    Outputs : TextAlignedAudioIU
    """

    @staticmethod
    def name():
        return "TTS DM Module"

    @staticmethod
    def description():
        return (
            "A module that synthesizes speech from text using coqui-ai's TTS library."
        )

    @staticmethod
    def input_ius():
        return [
            TurnTextIU,
            VADTurnAudioIU,
            DMIU,
        ]

    @staticmethod
    def output_iu():
        return TextAlignedAudioIU

    LANGUAGE_MAPPING = {
        "en": {
            "jenny": "tts_models/en/jenny/jenny",
            "vits": "tts_models/en/ljspeech/vits",
            "vits_neon": "tts_models/en/ljspeech/vits--neon",
            # "fast_pitch": "tts_models/en/ljspeech/fast_pitch", # bug sometimes
        },
        "multi": {
            "xtts_v2": "tts_models/multilingual/multi-dataset/xtts_v2",  # bugs
            "your_tts": "tts_models/multilingual/multi-dataset/your_tts",
            "vits_vctk": "tts_models/en/vctk/vits",
        },
    }

    def __init__(
        self,
        model="jenny",
        language="en",
        speaker_wav="TTS/wav_files/tts_api/tts_models_en_jenny_jenny/long_2.wav",
        frame_duration=0.2,
        verbose=False,
        device=None,
        **kwargs,
    ):
        """Initializes the CoquiTTSInterruption Module.

        Args:
            model (string): name of the desired model, has to be
                contained in the constant LANGUAGE_MAPPING.
            language (string): language of the desired model, has to be
                contained in the constant LANGUAGE_MAPPING.
            speaker_wav (string): path to a wav file containing the
                desired voice to copy (for voice cloning models).
            frame_duration (float): duration of the audio chunks
                contained in the outputted TextAlignedAudioIUs.
            verbose (bool, optional): the verbose level of the TTS
                model. Defaults to False.
            device (string, optional): the device the module will run on
                (cuda for gpu, or cpu)
        """
        super().__init__(**kwargs)

        # model
        if language not in self.LANGUAGE_MAPPING:
            print("Unknown TTS language. Defaulting to English (en).")
            language = "en"

        if model not in self.LANGUAGE_MAPPING[language].keys():
            print(
                "Unknown model for the following TTS language : "
                + language
                + ". Defaulting to "
                + next(iter(self.LANGUAGE_MAPPING[language]))
            )
            model = next(iter(self.LANGUAGE_MAPPING[language]))

        self.model = None
        self.model_name = self.LANGUAGE_MAPPING[language][model]
        self.device = device_definition(device)
        self.language = language
        self.speaker_wav = speaker_wav
        self.is_multilingual = language == "multi"

        # audio
        self.frame_duration = frame_duration
        self.samplerate = None
        self.samplewidth = 2
        self.chunk_size = None
        self.chunk_size_bytes = None

        # general
        self.verbose = verbose
        self._tts_thread_active = False
        self.iu_buffer = []
        self.buffer_pointer = 0
        self.interrupted_turn = -1
        self.current_turn_id = -1

        self.first_clause = True
        self.backchannel = None
        self.space_token = None

        self.bc_text = [
            "Yeah !",
            "okay",
            "alright",
            "Yeah, okay.",
            "uh",
            "uh, okay",
        ]

    def synthesize(self, text):
        """Takes the given text and synthesizes speech using the TTS model.
        Returns the synthesized speech as 22050 Hz int16-encoded numpy ndarray.

        Args:
            text (str): The text to use to synthesize speech.

        Returns:
            bytes: The speech as a 22050 Hz int16-encoded numpy ndarray.
        """

        final_outputs = self.model.tts(
            text=text,
            return_extra_outputs=True,
            split_sentences=False,
            verbose=self.verbose,
        )

        # if self.is_multilingual:
        #     # if "multilingual" in file or "vctk" in file:
        #     final_outputs = self.model.tts(
        #         text=text,
        #         language=self.language,
        #         speaker="p225",
        #         speaker_wav=self.speaker_wav,
        #         # speaker="Ana Florence",
        #     )
        # else:
        #     final_outputs = self.model.tts(text=text, speed=1.0)

        if len(final_outputs) != 2:
            raise NotImplementedError(
                "coqui TTS should output both wavforms and outputs"
            )
        else:
            waveforms, outputs = final_outputs

        # waveform = waveforms.squeeze(1).detach().numpy()[0]
        waveform = np.array(waveforms)

        # Convert float32 data [-1,1] to int16 data [-32767,32767]
        waveform = (waveform * 32767).astype(np.int16).tobytes()

        return waveform, outputs

    def one_clause_text_and_words(self, clause_ius):
        """Convert received IUs data accumulated in current_input list into a
        string.

        Returns:
            string: sentence chunk to synthesize speech from.
        """
        words = [iu.text for iu in clause_ius]
        return "".join(words), words

    def process_update(self, update_message):
        """Receives and stores the TurnTextIUs, so that they are later
        used to synthesize speech. Also receives useful top-level
        information about the dialogue from the DMIUS (interruptions,
        backchannels, etc).
        For now, the speech is synthsized every time a complete clause
        is received (with the _process_one_clause function).

        Args:
            update_message (UpdateType): UpdateMessage that contains new
                text to process, or top-level dialogue informations.

        Returns:
            _type_: returns None if update message is None.
        """
        if not update_message:
            return None

        clause_ius = []
        for iu, ut in update_message:
            if isinstance(iu, TurnTextIU):
                if iu.turn_id != self.interrupted_turn:
                    if ut == retico_core.UpdateType.ADD:
                        continue
                    elif ut == retico_core.UpdateType.REVOKE:
                        self.revoke(iu)
                    elif ut == retico_core.UpdateType.COMMIT:
                        clause_ius.append(iu)
            elif isinstance(iu, DMIU):
                if ut == retico_core.UpdateType.ADD:
                    if iu.action == "hard_interruption":
                        self.file_logger.info("hard_interruption")
                        self.interrupted_turn = self.current_turn_id
                        self.first_clause = True
                        self.current_input = []
                    elif iu.action == "soft_interruption":
                        self.file_logger.info("soft_interruption")
                    elif iu.action == "stop_turn_id":
                        self.terminal_logger.info(
                            "STOP TURN ID",
                            debug=True,
                            iu_turn=iu.turn_id,
                            curr=self.current_turn_id,
                        )
                        self.file_logger.info("stop_turn_id")
                        if iu.turn_id > self.current_turn_id:
                            self.interrupted_turn = self.current_turn_id
                        self.first_clause = True
                        self.current_input = []
                    elif iu.action == "back_channel":
                        self.terminal_logger.info("TTS BC", debug=True)
                        self.backchannel = self.bc_text[random.randint(0, 5)]
                    if iu.event == "user_BOT_same_turn":
                        self.interrupted_turn = None
                elif ut == retico_core.UpdateType.REVOKE:
                    continue
                elif ut == retico_core.UpdateType.COMMIT:
                    continue

        if len(clause_ius) != 0:
            self.current_input.append(clause_ius)

    def _process_one_clause(self):
        """function running in a separate thread, that synthesize and sends
        speech when a complete textual clause is received and appened in
        the current_input buffer."""
        while self._tts_thread_active:
            try:
                time.sleep(0.02)
                if len(self.current_input) != 0:
                    clause_ius = self.current_input.pop(0)
                    end_of_turn = clause_ius[-1].final
                    um = retico_core.UpdateMessage()
                    if end_of_turn:
                        self.terminal_logger.info(
                            "EOT TTS",
                            debug=True,
                            end_of_turn=end_of_turn,
                            clause_ius=clause_ius,
                        )
                        self.terminal_logger.info("EOT TTS")
                        self.file_logger.info("EOT")
                        self.first_clause = True
                        um.add_iu(
                            self.create_iu(grounded_in=clause_ius[-1], final=True),
                            retico_core.UpdateType.ADD,
                        )
                    else:
                        self.terminal_logger.info("EOC TTS")
                        if self.first_clause:
                            self.terminal_logger.info("start_answer_generation")
                            self.file_logger.info("start_answer_generation")
                            self.first_clause = False
                        self.current_turn_id = clause_ius[-1].turn_id
                        output_ius = self.get_new_iu_buffer_from_clause_ius(clause_ius)
                        um.add_ius(
                            [(iu, retico_core.UpdateType.ADD) for iu in output_ius]
                        )
                        self.file_logger.info("send_clause")
                    self.append(um)
                elif self.backchannel is not None:
                    um = retico_core.UpdateMessage()
                    output_ius = self.get_ius_backchannel()
                    um.add_ius([(iu, retico_core.UpdateType.ADD) for iu in output_ius])
                    self.append(um)
                    self.terminal_logger.info("TTS BC send_backchannel", debug=True)
                    self.file_logger.info("send_backchannel")
                    self.backchannel = None
            except Exception as e:
                log_exception(module=self, exception=e)

    def get_ius_backchannel(self):
        """Function that creates a list of BackchannelIUs containing audio that
        are the transcription of the chosen 'self.backchannel' string.

        Returns:
            list[BackchannelIU]: list of BackchannelIUs, transcriptions
                of 'self.backchannel'.
        """
        new_audio, outputs = self.synthesize(self.backchannel)
        outputs = outputs[0]
        len_wav = len(outputs["wav"])

        ius = []
        i = 0
        while i < len_wav:
            chunk = outputs["wav"][i : i + self.chunk_size]
            chunk = (np.array(chunk) * 32767).astype(np.int16).tobytes()
            if len(chunk) < self.chunk_size_bytes:
                chunk = chunk + b"\x00" * (self.chunk_size_bytes - len(chunk))

            i += self.chunk_size
            iu = BackchannelIU(
                creator=self,
                iuid=f"{hash(self)}:{self.iu_counter}",
                previous_iu=None,
                grounded_in=None,
                raw_audio=chunk,
                rate=self.samplerate,
                nframes=self.chunk_size,
                sample_width=self.samplewidth,
            )
            ius.append(iu)
        return ius

    def get_new_iu_buffer_from_clause_ius(self, clause_ius):
        """Function that aligns the TTS inputs and outputs. It links the words
        sent by LLM to audio chunks generated by TTS model. As we have access
        to the durations of the phonems generated by the model, we can link the
        audio chunks sent to speaker to the words that it corresponds to.

        Returns:
            list[TextAlignedAudioIU]: the TextAlignedAudioIUs that will
                be sent to the speaker module, containing the correct
                informations about grounded_iu, turn_id or char_id.
        """
        # preprocess on words
        current_text, words = self.one_clause_text_and_words(clause_ius)
        self.terminal_logger.info(
            "TTS get iu clause", debug=True, current_text=current_text, words=words
        )

        # pre_pro_words = []
        # pre_pro_words_distinct = []
        # try:
        #     for i, w in enumerate(words):
        #         if w[0] == " ":
        #             pre_pro_words.append(i - 1)
        #             if len(pre_pro_words) >= 2:
        #                 pre_pro_words_distinct.append(
        #                     words[pre_pro_words[-2] + 1 : pre_pro_words[-1] + 1]
        #                 )
        #             else:
        #                 pre_pro_words_distinct.append(words[: pre_pro_words[-1] + 1])
        #     self.terminal_logger.info(pre_pro_words, debug=True)
        #     self.terminal_logger.info(pre_pro_words_distinct, debug=True)
        #     pre_pro_words.pop(0)
        #     pre_pro_words_distinct.pop(0)
        #     pre_pro_words.append(len(words) - 1)
        # except IndexError as e:
        #     log_exception(self, e)
        #     raise IndexError from e

        # self.terminal_logger.info(pre_pro_words, debug=True)
        # self.terminal_logger.info(pre_pro_words_distinct, debug=True)

        # if len(pre_pro_words) >= 2:
        #     pre_pro_words_distinct.append(
        #         words[pre_pro_words[-2] + 1 : pre_pro_words[-1] + 1]
        #     )
        # else:
        #     pre_pro_words_distinct.append(words[: pre_pro_words[-1] + 1])

        npw = np.array(words)
        non_space_words = np.array([i for i, w in enumerate(npw) if w[0] != " "])
        pre_pro_words = list(np.delete(np.arange(len(npw)), non_space_words - 1))
        if len(pre_pro_words) == 0:
            pre_pro_words_distinct = []
        else:
            pre_pro_words_distinct = [words[0 : pre_pro_words[0] + 1]] + [
                words[x + 1 : pre_pro_words[i + 1] + 1]
                for i, x in enumerate(pre_pro_words[:-1])
            ]

        self.terminal_logger.info(pre_pro_words, debug=True)
        self.terminal_logger.info(pre_pro_words_distinct, debug=True)

        # hard coded values for the TTS model found in CoquiTTS github repo or calculated
        # SPACE_TOKEN_ID = 16
        # NB_FRAME_PER_DURATION = 256
        SPACE_TOKEN_ID = self.space_token
        NB_FRAME_PER_DURATION = 512

        self.file_logger.info("before_synthesize")
        new_audio, final_outputs = self.synthesize(current_text)
        self.file_logger.info("after_synthesize")
        tokens = self.model.synthesizer.tts_model.tokenizer.text_to_ids(current_text)
        self.file_logger.info("after_alignement")
        space_tokens_ids = []
        for i, x in enumerate(tokens):
            if x == SPACE_TOKEN_ID or i == len(tokens) - 1:
                space_tokens_ids.append(i + 1)

        # pre_tokenized_txt = [
        #     self.model.synthesizer.tts_model.tokenizer.decode([y]) for y in tokens
        # ]
        # pre_tokenized_text = [x if x != "<BLNK>" else "_" for x in pre_tokenized_txt]
        new_buffer = []
        for outputs in final_outputs:
            len_wav = len(outputs["wav"])
            durations = outputs["outputs"]["durations"].squeeze().tolist()
            # total_duration = int(sum(durations))

            wav_words_chunk_len = []
            old_len_w = 0
            for s_id in space_tokens_ids:
                wav_words_chunk_len.append(
                    int(sum(durations[old_len_w:s_id])) * NB_FRAME_PER_DURATION
                )
                # wav_words_chunk_len.append(int(sum(durations[old_len_w:s_id])) * len_wav / total_duration )
                old_len_w = s_id

            # if self.verbose:
            #     if len(pre_pro_words) > len(wav_words_chunk_len):
            #         print("TTS word alignment not exact, less tokens than words")
            #     elif len(pre_pro_words) < len(wav_words_chunk_len):
            #         print("TTS word alignment not exact, more tokens than words")

            cumsum_wav_words_chunk_len = list(np.cumsum(wav_words_chunk_len))

            i = 0
            j = 0
            while i < len_wav:
                chunk = outputs["wav"][i : i + self.chunk_size]
                # modify raw audio to match correct format
                chunk = (np.array(chunk) * 32767).astype(np.int16).tobytes()
                ## TODO : change that silence padding: padding with silence will slow down the speaker a lot
                word_id = pre_pro_words[-1]
                if len(chunk) <= self.chunk_size_bytes:
                    chunk = chunk + b"\x00" * (self.chunk_size_bytes - len(chunk))
                else:
                    while i + self.chunk_size >= cumsum_wav_words_chunk_len[j]:
                        j += 1
                    if j < len(pre_pro_words):
                        word_id = pre_pro_words[j]

                temp_word = words[word_id]
                grounded_iu = clause_ius[word_id]
                words_until_word_id = words[: word_id + 1]
                len_words = [len(word) for word in words[: word_id + 1]]
                char_id = sum(len_words) - 1

                i += self.chunk_size
                iu = self.create_iu(
                    grounded_in=grounded_iu,
                    audio=chunk,
                    chunk_size=self.chunk_size,
                    rate=self.samplerate,
                    sample_width=self.samplewidth,
                    grounded_word=temp_word,
                    word_id=word_id,
                    char_id=char_id,
                    turn_id=grounded_iu.turn_id,
                    clause_id=grounded_iu.clause_id,
                )
                new_buffer.append(iu)

        return new_buffer

    # def _tts_thread(self):
    #     """function used as a thread in the prepare_run function. Handles the messaging aspect of the retico module. if the clear_after_finish param is True,
    #     it means that speech chunks have been synthesized from a sentence chunk, and the speech chunks are sent to the children modules.
    #     """
    #     # TODO : change this function so that it sends the IUs without waiting for the IU duration to make it faster and let speaker module handle that ?
    #     # TODO : check if the usual system, like in the demo branch, works without this function, and having the message sending directly in process update function
    #     t1 = time.time()
    #     while self._tts_thread_active:
    #         try:
    #             t2 = t1
    #             t1 = time.time()
    #             if t1 - t2 < self.frame_duration:
    #                 time.sleep(self.frame_duration)
    #             else:
    #                 time.sleep(max((2 * self.frame_duration) - (t1 - t2), 0))

    #             if self.buffer_pointer >= len(self.iu_buffer):
    #                 self.buffer_pointer = 0
    #                 self.iu_buffer = []
    #             else:
    #                 iu = self.iu_buffer[self.buffer_pointer]
    #                 self.buffer_pointer += 1
    #                 um = retico_core.UpdateMessage.from_iu(
    #                     iu, retico_core.UpdateType.ADD
    #                 )
    #                 self.append(um)
    #         except Exception as e:
    #             log_exception(module=self, exception=e)

    def setup(self):
        super().setup()
        self.model = TTS(self.model_name).to(self.device)
        self.samplerate = self.model.synthesizer.tts_config.get("audio")["sample_rate"]
        self.chunk_size = int(self.samplerate * self.frame_duration)
        self.chunk_size_bytes = self.chunk_size * self.samplewidth
        self.space_token = self.model.synthesizer.tts_model.tokenizer.encode(" ")[0]

    def prepare_run(self):
        super().prepare_run()
        self.buffer_pointer = 0
        self.iu_buffer = []
        self._tts_thread_active = True
        # threading.Thread(target=self._tts_thread).start()
        threading.Thread(target=self._process_one_clause).start()

    def shutdown(self):
        super().shutdown()
        self._tts_thread_active = False
