import json
import threading
import time
import wave
from retico_amq import utils as amqu
import retico_core
from retico_core import log_utils

from retico_conversational_agent_unity.additional_IUs import DMIU, TextAlignedAudioIU


class GestureGeneratorModule(retico_core.abstract.AbstractModule):
    """A Module producing audio action from TextAlignedAudioIUs from TTS."""

    @staticmethod
    def name():
        return "GestureGenerator Module"

    @staticmethod
    def description():
        return "A Module producing audio action from TextAlignedAudioIUs from TTS."

    @staticmethod
    def input_ius():
        return [TextAlignedAudioIU, DMIU]

    @staticmethod
    def output_iu():
        return amqu.GestureIU

    def __init__(self, **kwargs):
        """
        Initialize the GestureGenerator Module.
        """
        super().__init__(**kwargs)
        self._thread_active = False
        self.cpt = 0
        self.audio_iu_buffer = []
        self.interrupted_turn_iu_buffer = []
        self.tts_framerate = 48000
        self.samplewidth = 2
        self.channels = 1

    def prepare_run(self):
        super().prepare_run()
        self._thread_active = True
        threading.Thread(target=self.run_process).start()

    def shutdown(self):
        super().shutdown()
        self._thread_active = False

    def process_update(self, update_message):
        clauses_ius = []
        for iu, ut in update_message:
            if isinstance(iu, DMIU):
                if ut == retico_core.UpdateType.ADD:
                    if iu.action == "continue":
                        self.terminal_logger.info("continue")
                        self.file_logger.info("continue")
                        output_iu = self.create_iu(
                            event="continue",
                        )
                        um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
                        self.append(um)
                        self.audio_iu_buffer = self.interrupted_turn_iu_buffer
                        self.soft_interrupted_iu = None

                    if iu.action == "soft_interruption":
                        self.terminal_logger.info(
                            "soft_interruption",
                            debug=True,
                            # grounded_word=self.latest_processed_iu.grounded_word,
                            # word_id=self.latest_processed_iu.word_id,
                            # char_id=self.latest_processed_iu.char_id,
                            clause_id=self.latest_processed_iu.clause_id,
                            turn_id=self.latest_processed_iu.turn_id,
                            final=iu.final,
                        )
                        self.file_logger.info("soft_interruption")
                        # if some iu was outputted, send to LLM module for alignement
                        if self.latest_processed_iu is not None:
                            output_iu = self.create_iu(
                                # grounded_word=self.latest_processed_iu.grounded_word,
                                # word_id=self.latest_processed_iu.word_id,
                                # char_id=self.latest_processed_iu.char_id,
                                clause_id=self.latest_processed_iu.clause_id,
                                turn_id=self.latest_processed_iu.turn_id,
                                final=iu.final,
                                event="interruption",
                            )
                            um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
                            self.append(um)
                            self.soft_interrupted_iu = output_iu
                            self.interrupted_turn_iu_buffer = self.audio_iu_buffer
                            self.audio_iu_buffer = []

                        else:
                            self.terminal_logger.info("speaker soft interruption but no outputted audio yet")
                            self.file_logger.info("speaker soft interruption but no outputted audio yet")

                    if iu.action == "hard_interruption":
                        self.terminal_logger.info("hard_interruption")
                        self.file_logger.info("hard_interruption")

                        # if some iu was outputted, send to LLM module for alignement
                        if self.latest_processed_iu is not None:
                            output_iu = self.create_iu(
                                # grounded_word=self.latest_processed_iu.grounded_word,
                                # word_id=self.latest_processed_iu.word_id,
                                # char_id=self.latest_processed_iu.char_id,
                                clause_id=self.latest_processed_iu.clause_id,
                                turn_id=self.latest_processed_iu.turn_id,
                                final=iu.final,
                                event="interruption",
                            )
                            um = retico_core.UpdateMessage()
                            um.add_ius(
                                [(um_iu, retico_core.UpdateType.ADD) for um_iu in self.current_output + [output_iu]]
                            )
                            self.append(um)
                            self.interrupted_iu = output_iu
                            # remove all audio in audio_buffer
                            self.audio_iu_buffer = []
                            self.current_output = []
                            # self.latest_processed_iu = None

                        else:
                            self.terminal_logger.info("speaker interruption but no outputted audio yet")
                            self.file_logger.info("speaker interruption but no outputted audio yet")

                    elif iu.event == "user_BOT_same_turn":
                        self.interrupted_iu = None

            if isinstance(iu, TextAlignedAudioIU):
                clauses_ius.append(iu)
        self.audio_iu_buffer.append(clauses_ius)

    def run_process(self):
        # The module doesn't send enough audio to have a continous signal, it's just a test module
        # If you want the module to send continous signal, change the time.sleep to time.sleep(self.frame_length), or the frame_length to 10
        while self._thread_active:
            if len(self.audio_iu_buffer) == 0:
                time.sleep(0.1)
            else:
                clause_ius = self.audio_iu_buffer.pop(0)
                if hasattr(clause_ius[0], "final") and clause_ius[0].final:
                    self.terminal_logger.info("agent_EOT")
                    self.file_logger.info("EOT")
                    output_iu = self.create_iu(
                        clause_id=clause_ius[0].clause_id,
                        turn_id=clause_ius[0].turn_id,
                        final=True,
                    )
                else:
                    output_iu = self.generate_gestures_one_clause(clause_ius)

                self.latest_processed_iu = output_iu
                um = retico_core.UpdateMessage()
                um.add_iu(output_iu, retico_core.UpdateType.ADD)
                self.append(um)
                self.terminal_logger.info(
                    "TestGestureDemo creates a retico IU",
                )

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
            #         "animation": "talking_4_shorter",
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

    def generate_gestures_one_clause(self, clause_ius):
        # recreate full audio
        full_data = b""
        full_sentence = ""
        for iu in clause_ius:
            full_data += bytes(iu.raw_audio)
            full_sentence += iu.grounded_word

        len_audio_bytes = len(full_data)
        len_audio_seconds = len_audio_bytes / (self.tts_framerate * self.samplewidth)

        self.terminal_logger.info(f"len_audio {len_audio_bytes} {len_audio_seconds} {full_sentence}", debug=True)

        # save full audio into wav file
        path = f"C:/Users/Sara Articulab/Documents/GitHub/retico_test/wav_files/clause_{clause_ius[0].clause_id}.wav"
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(self.channels)  # Set the number of channels
            wav_file.setsampwidth(self.samplewidth)  # Set the sample width in bytes
            wav_file.setframerate(self.tts_framerate)  # Set the frame rate (sample rate)
            wav_file.writeframes(full_data)  # Write the audio byte data

        # create audio action for AMQ
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
                "animation": "talking_4_shorter",
                "duration": len_audio_seconds,
                "delay": 0.0,
            },
        ]

        output_iu = self.create_iu(
            turnID=iu.turn_id, clauseID=iu.clause_id, interrupt=0, audios=audios, animations=animations
        )
        return output_iu

    def create_iu_from_dict(self, dict):
        return self.create_iu(**dict)

    def create_iu_from_json(self, path):
        with open(path, "rb") as f:
            data = json.load(f)
            return self.create_iu(**data)
