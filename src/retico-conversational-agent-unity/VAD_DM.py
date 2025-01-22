"""
VAD Module
=================

A retico module that provides Voice Activity Detection (VAD) using
WebRTC's VAD. Takes AudioIU as input, resamples the IU's raw_audio to
match WebRTC VAD's input frame rate, then call the VAD to predict
(user's) voice activity on the resampled raw_audio (True == speech
recognized), and finally returns the prediction alognside with the
raw_audio (and related parameter such as frame rate, etc) using a new IU
type called VADIU.

It also takes TextIU as input, to additionally keep tracks of the
agent's voice activity (agent == the retico system) by receiving IUs
from the SpeakerModule. The agent's voice activity is also outputted in
the VADIU.

Inputs : AudioIU, TextIU

Outputs : VADIU
"""

import pydub
import webrtcvad

import retico_core
from retico_core import audio

from retico_conversational_agent.additional_IUs import VADIU, SpeakerAlignementIU


class VadModule(retico_core.AbstractModule):
    """A retico module that provides Voice Activity Detection (VAD) using
    WebRTC's VAD. Takes AudioIU as input, resamples the IU's raw_audio to match
    WebRTC VAD's input frame rate, then call the VAD to predict (user's) voice
    activity on the resampled raw_audio (True == speech recognized), and
    finally returns the prediction alognside with the raw_audio (and related
    parameter such as frame rate, etc) using a new IU type called VADIU.

    It also takes SpeakerAlignementIU as input, to additionally keep
    tracks of the agent's voice activity (agent == the retico system) by
    receiving IUs from the SpeakerModule. The agent's voice activity is also
    outputted in the VADIU.
    """

    @staticmethod
    def name():
        return "VAD DM Module"

    @staticmethod
    def description():
        return "a module enhancing AudioIUs with voice activity for both user (using\
            WebRTC's VAD) and agent (using SpeakerAlignementIUs received from Speaker\
            Module)."

    @staticmethod
    def input_ius():
        return [audio.AudioIU, SpeakerAlignementIU]

    @staticmethod
    def output_iu():
        return VADIU

    def __init__(
        self,
        printing=False,
        target_framerate=16000,
        input_framerate=44100,
        channels=1,
        sample_width=2,
        vad_aggressiveness=3,
        **kwargs,
    ):
        """Initializes the VadModule Module.

        Args:
            target_framerate (int, optional): framerate of the output
                VADIUs (after resampling). Defaults to 16000.
            input_framerate (int, optional): framerate of the received
                AudioIUs. Defaults to 44100.
            channels (int, optional): number of channels (1=mono,
                2=stereo) of the received AudioIUs. Defaults to 1.
            sample_width (int, optional): sample width (number of bits
                used to encode each frame) of the received AudioIUs.
                Defaults to 2.
            vad_aggressiveness (int, optional): The level of
                aggressiveness of VAD model, the greater the more
                reactive. Defaults to 3.
        """
        super().__init__(**kwargs)
        self.printing = printing
        self.target_framerate = target_framerate
        self.input_framerate = input_framerate
        self.channels = channels
        self.sample_width = sample_width
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.VA_agent = False

    def resample_audio(self, raw_audio):
        """Resample the audio's frame_rate to correspond to
        self.target_framerate.

        Args:
            raw_audio (bytes): the audio received from the microphone that
                could need resampling.

        Returns:
            bytes: the resampled audio chunk.
        """
        if self.input_framerate != self.target_framerate:
            s = pydub.AudioSegment(
                raw_audio,
                sample_width=self.sample_width,
                channels=self.channels,
                frame_rate=self.input_framerate,
            )
            s = s.set_frame_rate(self.target_framerate)
            return s._data
        return raw_audio

    def process_update(self, update_message):
        """Receives SpeakerAlignementIU and AudioIU, use the first one to set the
        self.VA_agent class attribute, and process the second one by predicting
        whether it contains speech or not to set VA_user IU parameter.

        Args:
            update_message (UpdateType): UpdateMessage that contains new
                IUs (SpeakerAlignementIUs or AudioIUs), both are used to provide
                voice activity information (respectively for agent and
                user).
        """
        for iu, ut in update_message:
            # IUs from SpeakerModule, can be either agent BOT or EOT
            if isinstance(iu, SpeakerAlignementIU):
                if ut == retico_core.UpdateType.ADD:
                    # agent EOT
                    if iu.event == "agent_EOT":
                        self.VA_agent = False
                    if iu.event == "interruption":
                        self.VA_agent = False
                    # agent BOT
                    elif iu.event == "agent_BOT":
                        self.VA_agent = True
                    elif iu.event == "continue":
                        self.VA_agent = True
            elif isinstance(iu, audio.AudioIU):
                if ut == retico_core.UpdateType.ADD:
                    if self.input_framerate != iu.rate:
                        raise ValueError(
                            f"input framerate differs from iu framerate : {self.input_framerate} vs {iu.rate}"
                        )
                    raw_audio = self.resample_audio(iu.raw_audio)
                    VA_user = self.vad.is_speech(raw_audio, self.target_framerate)
                    output_iu = self.create_iu(
                        grounded_in=iu,
                        raw_audio=raw_audio,
                        nframes=iu.nframes,
                        rate=self.input_framerate,
                        sample_width=self.sample_width,
                        va_user=VA_user,
                        va_agent=self.VA_agent,
                    )
                    um = retico_core.UpdateMessage.from_iu(
                        output_iu, retico_core.UpdateType.ADD
                    )
                    self.append(um)

                    # something for logging
                    if self.VA_agent:
                        if VA_user:
                            event = "VA_overlap"
                        else:
                            event = "VA_agent"
                    else:
                        if VA_user:
                            event = "VA_user"
                        else:
                            event = "VA_silence"
                    self.file_logger.info(event)
