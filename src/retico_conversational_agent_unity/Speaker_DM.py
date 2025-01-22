"""
Speaker Module
==================

A retico module that outputs through the computer's speakers the audio contained in
TextAlignedAudioIUs. The module stops the speakers if it receives the information that the user
started talking (user barge-in/interruption of agent turn).
The interruption information is recognized by an VADTurnAudioIU with a parameter
vad_state="interruption".

The modules sends TextAlignedAudioIUs in 2 cases :
- When it is agent end of turn : when it consumes the last TextAlignedAudioIU of an agent turn
(with the parameter final=True), it sends back that IU to indicate to other modules that the
agent has stopped talking.
- When the agent turn is interrupted by the user : when it receives an VADTurnAudioIU with a
parameter vad_state="interruption", it sends the last TextAlignedAudioIU consumed (ie. the last
audio that has been spoken by the agent in the interrupted turn). It is useful to align the
dialogue history with the last spoken words.

Inputs : TextAlignedAudioIU, VADTurnAudioIU

Outputs : TextAlignedAudioIU
"""

import platform
import time
import pyaudio

import retico_core
import retico_core.abstract

from retico_conversational_agent.additional_IUs import (
    BackchannelIU,
    VADTurnAudioIU,
    TextAlignedAudioIU,
    DMIU,
    SpeakerAlignementIU,
)


class SpeakerDmModule(retico_core.AbstractModule):
    """A retico module that outputs through the computer's speakers the audio contained in
    TextAlignedAudioIUs. The module stops the speakers if it receives the information that the user
    started talking (user barge-in/interruption of agent turn).
    The interruption information is recognized by an VADTurnAudioIU with a parameter
    vad_state="interruption".

    The modules sends TextAlignedAudioIUs in 2 cases :
    - When it is agent end of turn : when it consumes the last TextAlignedAudioIU of an agent turn
    (with the parameter final=True), it sends back that IU to indicate to other modules that the
    agent has stopped talking.
    - When the agent turn is interrupted by the user : when it receives an VADTurnAudioIU with a
    parameter vad_state="interruption", it sends the last TextAlignedAudioIU consumed (ie. the last
    audio that has been spoken by the agent in the interrupted turn). It is useful to align the
    dialogue history with the last spoken words.

    Inputs : TextAlignedAudioIU, VADTurnAudioIU

    Outputs : TextAlignedAudioIU
    """

    @staticmethod
    def name():
        return "Speaker DM Module"

    @staticmethod
    def description():
        return "A module that plays audio to speakers and stops playing audio if the user starts speaking."

    @staticmethod
    def input_ius():
        return [
            TextAlignedAudioIU,
            VADTurnAudioIU,
            DMIU,
            BackchannelIU,
            retico_core.audio.AudioIU,
        ]

    @staticmethod
    def output_iu():
        return SpeakerAlignementIU

    def __init__(
        self,
        rate=44100,
        frame_length=0.2,
        channels=1,
        sample_width=2,
        use_speaker="both",
        device_index=None,
        **kwargs,
    ):
        """
        Initializes the SpeakerInterruption Module.

        Args:
            rate (int): framerate of the played audio chunks.
            frame_length (float): duration of the played audio chunks.
            channels (int): number of channels (1=mono, 2=stereo) of the received VADTurnAudioIUs.
            sample_width (int):sample width (number of bits used to encode each frame) of the received VADTurnAudioIUs.
            use_speaker (string): wether the audio should be played in the right, left or both speakers.
            device_index(string):
        """
        super().__init__(**kwargs)
        self.rate = rate
        self.sample_width = sample_width
        self.use_speaker = use_speaker
        self.channels = channels
        self._p = pyaudio.PyAudio()
        if device_index is None:
            device_index = self._p.get_default_output_device_info()["index"]
        self.device_index = device_index
        self.stream = None

        # interruptions parameters
        self.audio_iu_buffer = []
        self.latest_processed_iu = None
        self.interrupted_iu = None
        self.soft_interrupted_iu = None
        self.interrupted_turn_iu_buffer = []
        self.frame_length = frame_length

        self.backchannel_iu_buffer = []

    def process_update(self, update_message):
        """overrides SpeakerModule : https://github.com/retico-team/retico-core/blob/main/retico_core/audio.py#L282

        overrides SpeakerModule's process_update to save logs.
        """
        for iu, ut in update_message:
            if isinstance(iu, DMIU):
                if ut == retico_core.UpdateType.ADD:
                    if iu.action == "continue":
                        self.terminal_logger.info("continue")
                        self.file_logger.info("continue")
                        output_iu = self.create_iu(
                            event="continue",
                        )
                        um = retico_core.UpdateMessage.from_iu(
                            output_iu, retico_core.UpdateType.ADD
                        )
                        self.append(um)
                        self.audio_iu_buffer = self.interrupted_turn_iu_buffer
                        self.soft_interrupted_iu = None
                    if iu.action == "soft_interruption":
                        self.terminal_logger.info(
                            "soft_interruption",
                            debug=True,
                            grounded_word=self.latest_processed_iu.grounded_word,
                            word_id=self.latest_processed_iu.word_id,
                            char_id=self.latest_processed_iu.char_id,
                            clause_id=self.latest_processed_iu.clause_id,
                            turn_id=self.latest_processed_iu.turn_id,
                            final=iu.final,
                        )
                        self.file_logger.info("soft_interruption")
                        # if some iu was outputted, send to LLM module for alignement
                        if self.latest_processed_iu is not None:
                            output_iu = self.create_iu(
                                grounded_word=self.latest_processed_iu.grounded_word,
                                word_id=self.latest_processed_iu.word_id,
                                char_id=self.latest_processed_iu.char_id,
                                clause_id=self.latest_processed_iu.clause_id,
                                turn_id=self.latest_processed_iu.turn_id,
                                final=iu.final,
                                event="interruption",
                            )
                            um = retico_core.UpdateMessage.from_iu(
                                output_iu, retico_core.UpdateType.ADD
                            )
                            self.append(um)
                            self.soft_interrupted_iu = output_iu
                            self.interrupted_turn_iu_buffer = self.audio_iu_buffer
                            self.audio_iu_buffer = []
                            self.current_output = []
                        else:
                            self.terminal_logger.info(
                                "speaker soft interruption but no outputted audio yet"
                            )
                            self.file_logger.info(
                                "speaker soft interruption but no outputted audio yet"
                            )
                    elif iu.action == "hard_interruption":
                        self.terminal_logger.info("hard_interruption")
                        self.file_logger.info("hard_interruption")
                        # if some iu was outputted, send to LLM module for alignement
                        if self.latest_processed_iu is not None:
                            output_iu = self.create_iu(
                                grounded_word=self.latest_processed_iu.grounded_word,
                                word_id=self.latest_processed_iu.word_id,
                                char_id=self.latest_processed_iu.char_id,
                                clause_id=self.latest_processed_iu.clause_id,
                                turn_id=self.latest_processed_iu.turn_id,
                                final=iu.final,
                                event="interruption",
                            )
                            um = retico_core.UpdateMessage()
                            um.add_ius(
                                [
                                    (um_iu, retico_core.UpdateType.ADD)
                                    for um_iu in self.current_output + [output_iu]
                                ]
                            )
                            self.append(um)
                            self.interrupted_iu = output_iu
                            # remove all audio in audio_buffer
                            self.audio_iu_buffer = []
                            self.current_output = []
                            # self.latest_processed_iu = None
                        else:
                            self.terminal_logger.info(
                                "speaker interruption but no outputted audio yet"
                            )
                            self.file_logger.info(
                                "speaker interruption but no outputted audio yet"
                            )
                    elif iu.action == "repeat_last_turn":
                        self.terminal_logger.info("repeat ius received", debug=True)
                        self.audio_iu_buffer.append(iu)

                    elif iu.event == "user_BOT_same_turn":
                        self.interrupted_iu = None

            elif isinstance(iu, TextAlignedAudioIU):
                if ut == retico_core.UpdateType.ADD:
                    if self.interrupted_iu is not None:
                        # if, after an interrupted turn, an IU from a new turn has been received
                        if not iu.final and self.interrupted_iu.turn_id != iu.turn_id:
                            self.interrupted_iu = None
                            self.audio_iu_buffer.append(iu)
                    elif self.soft_interrupted_iu is not None:
                        self.terminal_logger.info(
                            "IU received during soft interruption",
                            debug=True,
                            soft_inter_iu_turn=self.soft_interrupted_iu.turn_id,
                            TTS_iu_turn=iu.turn_id,
                            iu_final=iu.final,
                        )
                        if (
                            not iu.final
                            and self.soft_interrupted_iu.turn_id != iu.turn_id
                        ):
                            self.soft_interrupted_iu = None
                            self.audio_iu_buffer.append(iu)
                            self.interrupted_turn_iu_buffer = []
                        else:
                            self.interrupted_turn_iu_buffer.append(iu)
                    else:
                        self.audio_iu_buffer.append(iu)
            elif isinstance(iu, BackchannelIU):
                self.backchannel_iu_buffer.append(iu)

        return None

    def callback(self, in_data, frame_count, time_info, status):
        """callback function given to the pyaudio stream that will output audio to the computer speakers.
        This function returns an audio chunk that will be written in the stream.
        It is called everytime the last chunk has been fully consumed.

        Args:
            in_data (_type_): _description_
            frame_count (_type_): number of frames in an audio chunk written in the stream
            time_info (_type_): _description_
            status (_type_): _description_

        Returns:
            (bytes, pyaudio type): the tuple containing the audio chunks (bytes)
            and the pyaudio type informing wether the stream should continue or stop.
        """
        if len(self.backchannel_iu_buffer) > 0:
            iu = self.backchannel_iu_buffer.pop(0)
            data = bytes(iu.raw_audio)
            return (data, pyaudio.paContinue)
        if len(self.audio_iu_buffer) == 0:
            self.terminal_logger.info("output_silence")
            self.file_logger.info("output_silence")
            silence_bytes = b"\x00" * frame_count * self.channels * self.sample_width
            return (silence_bytes, pyaudio.paContinue)

        iu = self.audio_iu_buffer.pop(0)
        # if it is the last IU from TTS for this agent turn, which corresponds to an agent EOT.
        if hasattr(iu, "final") and iu.final:
            self.terminal_logger.info("agent_EOT")
            self.file_logger.info("EOT")
            output_iu = self.create_iu(
                grounded_word=iu.grounded_word,
                word_id=iu.word_id,
                char_id=iu.char_id,
                clause_id=iu.clause_id,
                turn_id=iu.turn_id,
                final=iu.final,
                event="agent_EOT",
            )

            um = retico_core.UpdateMessage()
            um.add_ius(
                [
                    (um_iu, retico_core.UpdateType.ADD)
                    for um_iu in self.current_output + [output_iu]
                ]
            )
            self.current_output = []
            self.append(um)
            silence_bytes = b"\x00" * frame_count * self.channels * self.sample_width
            return (silence_bytes, pyaudio.paContinue)
        else:
            # if it is the first IU from new agent turn, which corresponds to the official agent BOT
            if self.latest_processed_iu is None or (
                self.latest_processed_iu.turn_id is not None
                and iu.turn_id is not None
                and self.latest_processed_iu.turn_id != iu.turn_id
            ):
                self.terminal_logger.info("agent_BOT")
                self.file_logger.info("agent_BOT")
                output_iu = self.create_iu(
                    grounded_word=iu.grounded_word,
                    word_id=iu.word_id,
                    char_id=iu.char_id,
                    clause_id=iu.clause_id,
                    turn_id=iu.turn_id,
                    final=iu.final,
                    event="agent_BOT",
                )
                um = retico_core.UpdateMessage.from_iu(
                    output_iu, retico_core.UpdateType.ADD
                )
                self.append(um)

            self.terminal_logger.info("output_audio")
            self.file_logger.info("output_audio")
            data = bytes(iu.raw_audio)
            self.latest_processed_iu = iu
            stored_iu = self.create_iu(
                raw_audio=iu.raw_audio,
                nframes=iu.nframes,
                rate=iu.rate,
                sample_width=iu.sample_width,
                grounded_word=iu.grounded_word,
                word_id=iu.word_id,
                char_id=iu.char_id,
                clause_id=iu.clause_id,
                turn_id=iu.turn_id,
                final=iu.final,
                event="ius_from_last_turn",
            )
            self.current_output.append(stored_iu)
            return (data, pyaudio.paContinue)

    def prepare_run(self):
        """Open the stream to enable sound outputting through speakers"""
        super().prepare_run()
        p = self._p

        if platform.system() == "Darwin":
            if self.use_speaker == "left":
                stream_info = pyaudio.PaMacCoreStreamInfo(channel_map=(0, -1))
            elif self.use_speaker == "right":
                stream_info = pyaudio.PaMacCoreStreamInfo(channel_map=(-1, 0))
            else:
                stream_info = pyaudio.PaMacCoreStreamInfo(channel_map=(0, 0))
        else:
            stream_info = None

        # Adding the stream_callback parameter should make the stream.write() function non blocking,
        # which would make it possible to run in parallel of the reception of update messages (and the uptdate of vad_state)
        self.stream = p.open(
            format=p.get_format_from_width(self.sample_width),
            channels=self.channels,
            rate=self.rate,
            input=False,
            output_host_api_specific_stream_info=stream_info,
            output=True,
            output_device_index=self.device_index,
            stream_callback=self.callback,
            frames_per_buffer=int(self.rate * self.frame_length),
        )

        self.stream.start_stream()

    def shutdown(self):
        """Close the audio stream."""
        super().shutdown()
        self.stream.stop_stream()
        self.stream.close()
        self.stream = None
