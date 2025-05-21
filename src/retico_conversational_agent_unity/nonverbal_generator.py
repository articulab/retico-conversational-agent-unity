import io
import json
import os
import pathlib
import threading
import time
import wave

import retico_core
from retico_amq import GestureIU
from retico_conversational_agent import DMIU, TextAlignedAudioIU


class NonverbalGeneratorModule(retico_core.abstract.AbstractModule):
    """A Module producing audio action from TextAlignedAudioIUs from TTS."""

    @staticmethod
    def name():
        return "NonverbalGenerator Module"

    @staticmethod
    def description():
        return "A Module producing audio action from TextAlignedAudioIUs from TTS."

    @staticmethod
    def input_ius():
        return [TextAlignedAudioIU, DMIU]

    @staticmethod
    def output_iu():
        return GestureIU

    def __init__(
        self,
        tts_framerate=48000,
        samplewidth=2,
        channels=1,
        store_audio=False,
        **kwargs,
    ):
        """
        Initialize the NonverbalGenerator Module.
        """
        super().__init__(**kwargs)
        self._thread_active = False
        self.cpt = 0
        self.clause_ius_buffer = []
        self.tts_framerate = tts_framerate
        self.samplewidth = samplewidth
        self.channels = channels
        self.first_clause = True
        self.interrupted_turn = -1
        self.current_turn_id = -1
        self.store_audio = store_audio

    def prepare_run(self):
        super().prepare_run()
        self._thread_active = True
        threading.Thread(target=self._nvg_thread).start()

    def shutdown(self):
        super().shutdown()
        self._thread_active = False

    def process_update(self, update_message):
        clause_ius = []
        for iu, ut in update_message:
            if isinstance(iu, TextAlignedAudioIU):
                if iu.turn_id != self.interrupted_turn:
                    clause_ius.append(iu)  # ? Append the clauses to buffer
            if isinstance(iu, DMIU):
                if ut == retico_core.UpdateType.ADD:
                    if iu.action == "hard_interruption":
                        self.file_logger.info("hard_interruption")
                        self.interrupted_turn = self.current_turn_id
                        self.first_clause = True
                        self.clause_ius_buffer = []
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
                        self.clause_ius_buffer = []
                    if iu.event == "user_BOT_same_turn":
                        self.interrupted_turn = None
        if len(clause_ius) != 0:
            self.clause_ius_buffer.append(clause_ius)

    def _nvg_thread(self):
        # The module doesn't send enough audio to have a continous signal, it's just a test module
        # If you want the module to send continous signal, change the time.sleep to time.sleep(self.frame_length), or the frame_length to 10
        while self._thread_active:
            if len(self.clause_ius_buffer) == 0:
                time.sleep(0.1)
            else:
                clause_ius = self.clause_ius_buffer.pop(0)
                if hasattr(clause_ius[0], "final") and clause_ius[0].final:
                    self.terminal_logger.info("agent_EOT")
                    self.file_logger.info("EOT")
                    output_iu = self.create_iu(
                        turnID=clause_ius[0].turn_id,
                        final=True,
                    )
                    self.first_clause = True
                else:
                    self.terminal_logger.info("EOC NV")
                    if self.first_clause:
                        self.terminal_logger.info("start_answer_generation")
                        self.file_logger.info("start_answer_generation")
                        self.first_clause = False
                    self.current_turn_id = clause_ius[-1].turn_id
                    if self.store_audio:
                        output_iu = self.generate_nonverbal_one_clause_audio_file(
                            clause_ius
                        )
                    else:
                        output_iu = self.generate_nonverbal_one_clause_audio_bytes(
                            clause_ius
                        )
                    self.file_logger.info("send_clause")

                um = retico_core.UpdateMessage()
                um.add_iu(output_iu, retico_core.UpdateType.ADD)
                self.append(um)
                self.terminal_logger.info(
                    "NonverbalGenerator creates a retico IU",
                )

    def generate_nonverbal_one_clause_audio_file(self, clause_ius):
        # recreate full audio
        full_data = b""
        full_sentence = ""
        for iu in clause_ius:
            full_data += bytes(iu.raw_audio)
            full_sentence += iu.grounded_word
        len_audio_bytes = len(full_data)
        len_audio_seconds = len_audio_bytes / (self.tts_framerate * self.samplewidth)
        # self.terminal_logger.info(f"len_audio {len_audio_bytes} {len_audio_seconds} {full_sentence}", debug=True)

        # save full audio into wav file
        current_local_path = pathlib.Path(__file__).parent.resolve()
        folder_path = f"{current_local_path}/wav_files/"
        filename = f"clause_{clause_ius[0].clause_id}.wav"
        path = folder_path + filename
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(self.channels)  # Set the number of channels
            wav_file.setsampwidth(self.samplewidth)  # Set the sample width in bytes
            wav_file.setframerate(
                self.tts_framerate
            )  # Set the frame rate (sample rate)
            wav_file.writeframes(full_data)  # Write the audio byte data

        # create audio action for AMQ
        interrupt = 2
        audios = [
            {
                "path": path,
                "transcription": "TEST DEMO",
                "volume": 1,
                # "delay": 0,
                # "Timing Index": 0
            },
        ]
        animations = [
            {
                "animation": "talking_4",
                "duration": len_audio_seconds,
                "delay": 0.0,
            },
        ]
        output_iu = self.create_iu(
            interrupt=interrupt,
            turnID=iu.turn_id,
            clauseID=iu.clause_id,
            audios=audios,
            animations=animations,
        )
        return output_iu

    def generate_nonverbal_one_clause_audio_bytes(self, clause_ius):
        # recreate full audio
        full_data = b""
        full_sentence = ""
        for iu in clause_ius:
            full_data += bytes(iu.raw_audio)
            full_sentence += iu.grounded_word
        len_audio_bytes = len(full_data)
        len_audio_seconds = len_audio_bytes / (self.tts_framerate * self.samplewidth)
        # self.terminal_logger.info(f"len_audio {len_audio_bytes} {len_audio_seconds} {full_sentence}", debug=True)

        # convert audio_bytes to make it possible to play in Unity
        full_data = retico_core.audio.convert_audio_PCM16_to_WAVPCM16(
            raw_audio=full_data,
            sample_rate=(
                clause_ius[0].rate if clause_ius[0].rate else self.tts_framerate
            ),
            num_channels=self.channels,
            sampwidth=(
                clause_ius[0].sample_width
                if clause_ius[0].sample_width
                else self.samplewidth
            ),
        )

        # create audio action for AMQ
        interrupt = 2
        audios = [
            {
                "bytes": full_data,
                "transcription": "TEST DEMO",
                "volume": 1,
                # "delay": 0,
                # "Timing Index": 0
            },
        ]
        animations = [
            {
                "animation": "talking_4",
                "duration": len_audio_seconds,
                "delay": 0.0,
            },
        ]
        output_iu = self.create_iu(
            interrupt=interrupt,
            turnID=iu.turn_id,
            clauseID=iu.clause_id,
            audios=audios,
            animations=animations,
        )
        return output_iu

    def create_iu_from_dict(self, dict):
        return self.create_iu(**dict)

    def create_iu_from_json(self, path):
        with open(path, "rb") as f:
            data = json.load(f)
            return self.create_iu(**data)

            # In generate_nonverbal_one_clause
            # try:
            # turnID = self.cpt // 2
            # clauseID = self.cpt % 2
            # interrupt = 0
            # timings = [0, 0.384]
            # audios = [
            #     {
            #         "path": "C:/Users/Sara Articulab/Documents/GitHub/simple-retico-agent/src/audio_0.wav",
            #         "transcription" : "Hello,",
            #         "volume": 1,
            #         "delay": 1,
            #         "Timing Index": 0
            #     },
            #     {
            #         "path": "C:/Users/Sara Articulab/Documents/GitHub/simple-retico-agent/src/audio_1.wav",
            #         "transcription" : "My name is Marius!",
            #         "volume": 1,
            #         "delay": 1,
            #         "Timing Index": 1
            #     }
            # ]
            # animations = [
            #     {
            #         "animation": "greeting_waiving_shorter",
            #         "bodypart": "rightarm",
            #         "duration": 0.0,
            #         "delay": 0.0,
            #     },
            #     {
            #         "animation": "talking_4",
            #         # "bodypart": "all",
            #         "duration": 3.0,
            #         "delay": 0.0,
            #     },
            # ]
            # blendshapes = [
            #     {
            #         "id": "A38_Mouth_Smile_Left",
            #         "value": 0.22,
            #         "duration": 1.7,
            #         "delay": 0.5,
            #     },
            #     {
            #         "id": "A39_Mouth_Smile_Right",
            #         "value": 0.31,
            #         "duration": 1.7,
            #         "delay": 0.5,
            #     },
            #     {
            #         "id": "A42_Mouth_Dimple_Left",
            #         "value": 0.25,
            #         "duration": 1.7,
            #         "delay": 0.5,
            #     },
            #     {
            #         "id": "A43_Mouth_Dimple_Right",
            #         "value": 0.12,
            #         "duration": 1.7,
            #         "delay": 0.5,
            #     },
            #     # {"id": "sad", "value": 1.0, "duration": 1.0, "delay": 1.0},
            # ]
            # # lookAt = [{"x": 0, "y": 0, "z": 0, "duration": 2.0, "delay": 0.0}]
            # # gazes = [
            # #     {"x": 30, "y": 50, "duration": 1.0, "delay": 0.0},
            # #     {"x": 0, "y": 0, "duration": 1.0, "delay": 1.0},
            # # ]
            # # left_hand_movements = [
            # #     {"x": 100, "y": 30, "duration": 1.0, "delay": 1.0}
            # # ]
            # # right_hand_movements = [
            # #     {"x": 30, "y": 0, "duration": 0.5, "delay": 0.0},
            # #     {"x": 0, "y": 50, "duration": 1.0, "delay": 0.5},
            # # ]

            # iu = self.create_iu(
            #     turnID=turnID,
            #     clauseID=clauseID,
            #     interrupt=interrupt,
            #     animations=animations,
            #     blendshapes=blendshapes,
            #     audios=audios,
            #     timings=timings,
            #     # lookAt=lookAt,
            #     # gazes=gazes,
            #     # left_hand_movements=left_hand_movements,
            #     # right_hand_movements=right_hand_movements,

            # )

            #     iu = self.create_iu_from_json("greeting_demo.json")

            # um = retico_core.UpdateMessage()
            # um.add_iu(iu, retico_core.UpdateType.ADD)
            # self.append(um)
            # self.terminal_logger.info(
            #     "TestProducingModule creates a retico IU",
            # )
            # self.cpt += 1
            # time.sleep(30)
            # except Exception as e:
            #     log_utils.log_exception(module=self, exception=e)
