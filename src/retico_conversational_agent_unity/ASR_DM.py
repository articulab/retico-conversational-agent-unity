"""
Whisper ASR Module
==================

A retico module that provides Automatic Speech Recognition (ASR) using a
OpenAI's Whisper model. Periodically predicts a new text hypothesis from
the input incremental speech and predicts a final hypothesis when it is
the user end of turn.

The received VADTurnAudioIU are stored in a buffer from which a
prediction is made periodically, the words that were not present in the
previous hypothesis are ADDED, in contrary, the words that were present,
but aren't anymore are REVOKED. It recognize the user's EOT information
when COMMIT VADTurnAudioIUs are received, a final prediciton is then
made and the corresponding IUs are COMMITED.

The faster_whisper library is used to speed up the whisper inference.

Inputs : VADTurnAudioIU

Outputs : SpeechRecognitionIU
"""

import os
import threading
import time
import numpy as np
import transformers
from faster_whisper import WhisperModel

import retico_core
from retico_core.log_utils import log_exception

from retico_conversational_agent.utils import device_definition
from retico_conversational_agent.additional_IUs import DMIU, SpeechRecognitionTurnIU

transformers.logging.set_verbosity_error()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class AsrDmModule(retico_core.AbstractModule):
    """A retico module that provides Automatic Speech Recognition (ASR) using a
    OpenAI's Whisper model. Periodically predicts a new text hypothesis from
    the input incremental speech and predicts a final hypothesis when it is the
    user end of turn.

    The received VADTurnAudioIU are stored in a buffer from which a
    prediction is made periodically, the words that were not present in
    the previous hypothesis are ADDED, in contrary, the words that were
    present, but aren't anymore are REVOKED. It recognize the user's EOT
    information when COMMIT VADTurnAudioIUs are received, a final
    prediciton is then made and the corresponding IUs are COMMITED.

    The faster_whisper library is used to speed up the whisper
    inference.

    Inputs : VADTurnAudioIU

    Outputs : SpeechRecognitionTurnIU
    """

    @staticmethod
    def name():
        return "ASR Whisper DM Module"

    @staticmethod
    def description():
        return "A module that recognizes transcriptions from speech using Whisper."

    @staticmethod
    def input_ius():
        return [DMIU]

    @staticmethod
    def output_iu():
        return SpeechRecognitionTurnIU

    def __init__(
        self,
        whisper_model="distil-large-v2",
        device=None,
        framerate=16000,
        **kwargs,
    ):
        """Initializes the WhisperASRInterruption Module.

        Args:
            whisper_model (string): name of the desired model, has to
                correspond to a model in the faster_whisper library.
            device (string): wether the model will be executed on cpu or
                gpu (using "cuda").
            framerate (int, optional): framerate of the received VADIUs.
                Defaults to 16000.
        """
        super().__init__(**kwargs)

        # model
        self.device = device_definition(device)
        self.model = WhisperModel(
            whisper_model, device=self.device, compute_type="int8"
        )

        # general
        self._asr_thread_active = False
        self.latest_input_iu = None
        self.eos = False
        self.audio_buffer = []

        # audio
        self.framerate = framerate

    def recognize(self):
        """Recreate the audio signal received by the microphone by
        concatenating the audio chunks from the audio_buffer and transcribe
        this concatenation into a list of predicted words.

        Returns:
            (list[string], boolean): the list of transcribed words.
        """
        # faster whisper
        full_audio = b"".join(self.audio_buffer)
        audio_np = (
            np.frombuffer(full_audio, dtype=np.int16).astype(np.float32) / 32768.0
        )
        segments, _ = self.model.transcribe(audio_np)  # the segments can be streamed
        segments = list(segments)
        transcription = "".join([s.text for s in segments])

        return transcription

    def process_update(self, update_message):
        """Receives and stores the audio from the DMIUs in the
        self.audio_buffer buffer.

        Args:
            update_message (UpdateType): UpdateMessage that contains new
                IUs, if their UpdateType is ADD, they are added to the
                audio_buffer.
        """
        eos = False
        for iu, ut in update_message:
            if iu.action == "process_audio":
                if self.framerate != iu.rate:
                    raise ValueError("input framerate differs from iu framerate")
                # ADD corresponds to new audio chunks of user sentence, to generate new transcription hypothesis
                if ut == retico_core.UpdateType.COMMIT:
                    self.terminal_logger.info("start_process")
                    self.file_logger.info("start_process")
                    eos = True
                    self.audio_buffer.append(iu.raw_audio)
                    if not self.latest_input_iu:
                        self.latest_input_iu = iu
        if eos:
            self.eos = eos

    def _asr_thread(self):
        """Function used as a thread in the prepare_run function. Handles the
        messaging aspect of the retico module. Calls the Whisper model to
        generate a prediction from the audio contained in the audio_buffer
        sub-class's. ADD the new words and COMMITS the final prediction.

        (Only called at the user EOT for now).
        """
        while self._asr_thread_active:
            try:
                time.sleep(0.01)
                if not self.eos:
                    continue

                prediction = self.recognize()
                self.file_logger.info("predict")
                if len(prediction) != 0:
                    um, new_tokens = retico_core.text.get_text_increment(
                        self, prediction
                    )
                    for i, token in enumerate(new_tokens):
                        output_iu = self.create_iu(
                            grounded_in=self.latest_input_iu,
                            predictions=[prediction],
                            text=token,
                            stability=0.0,
                            confidence=0.99,
                            final=self.eos and (i == (len(new_tokens) - 1)),
                            turn_id=self.latest_input_iu.turn_id,
                        )
                        um.add_iu(output_iu, retico_core.UpdateType.ADD)
                        self.commit(output_iu)
                        um.add_iu(output_iu, retico_core.UpdateType.COMMIT)

                    self.audio_buffer = []
                    self.eos = False
                    self.latest_input_iu = None
                    self.file_logger.info("send_clause")
                    self.append(um)
            except Exception as e:
                log_exception(module=self, exception=e)

    def prepare_run(self):
        super().prepare_run()
        self._asr_thread_active = True
        threading.Thread(target=self._asr_thread).start()

    def shutdown(self):
        super().shutdown()
        self._asr_thread_active = False
