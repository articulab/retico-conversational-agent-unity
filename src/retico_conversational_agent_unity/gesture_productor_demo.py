import json
import threading
import time
import wave
from retico_amq import utils as amqu
import retico_core
from retico_core import log_utils

from retico_conversational_agent_unity.additional_IUs import TextAlignedAudioIU


class GestureProducerDemoModule(retico_core.abstract.AbstractProducingModule):

    @staticmethod
    def name():
        return "GestureProducerDemo Module"

    @staticmethod
    def description():
        return "A Module producing non-verbal behavior every n seconds."

    @staticmethod
    def input_ius():
        return []

    @staticmethod
    def output_iu():
        return amqu.GestureIU

    def __init__(self, **kwargs):
        """
        Initialize the GestureProducerDemo Module.
        """
        super().__init__(**kwargs)
        self._thread_active = False
        self.cpt = 0
        self.cpt_2 = 0

    def prepare_run(self):
        super().prepare_run()
        self._thread_active = True
        threading.Thread(target=self.run_process).start()

    def shutdown(self):
        super().shutdown()
        self._thread_active = False

    def process_update(self, update_message):
        pass

    def run_process(self):
        # The module doesn't send enough audio to have a continous signal, it's just a test module
        # If you want the module to send continous signal, change the time.sleep to time.sleep(self.frame_length), or the frame_length to 10
        while self._thread_active:
            animations = [
                {
                    "animation": "talking_4_shorter",
                    # "duration": 0,
                    # "delay": 0.0,
                },
            ]

            output_iu = self.create_iu(
                turnID=self.cpt_2, clauseID=self.cpt, interrupt=0, animations=animations, final=(self.cpt == 4)
            )
            um = retico_core.UpdateMessage()
            um.add_iu(output_iu, retico_core.UpdateType.ADD)
            self.append(um)
            self.terminal_logger.info("GestureProducerDemo creates a retico IU", final=output_iu.final)

            self.cpt += 1
            if self.cpt == 5:
                self.cpt_2 += 1
                self.cpt = 0
                time.sleep(10)
            else:
                time.sleep(2)

    def create_iu_from_dict(self, dict):
        return self.create_iu(**dict)

    def create_iu_from_json(self, path):
        with open(path, "rb") as f:
            data = json.load(f)
            return self.create_iu(**data)
