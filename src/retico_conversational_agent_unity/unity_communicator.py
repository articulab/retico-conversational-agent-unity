import threading
import time

import retico_core
from retico_amq import GestureIU
from retico_conversational_agent import DMIU, SpeakerAlignementIU
from .additional_IUs import UnityMessageIU


class UnityCommunicatorModule(retico_core.abstract.AbstractModule):
    @staticmethod
    def name():
        return "UnityCommunicator Module"

    @staticmethod
    def description():
        return "A module that sends and receives IUs to the Unity side."

    @staticmethod
    def input_ius():
        return [DMIU, UnityMessageIU, GestureIU]

    @staticmethod
    def output_iu():
        return retico_core.abstract.IncrementalUnit  # SpeakerAlignementIU, amqu.GestureIU

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._thread_active = False
        self.last_clause_each_turn = dict()
        self.last_clause_each_turn_temp = dict()
        self.last_command_started_but_not_ended = None
        self.first_clause = True
        self.interrupted_iu = None
        self.soft_interrupted_iu = None
        self.interrupted_turn_iu_buffer = []
        self.last_command_ended = None

    def prepare_run(self):
        super().prepare_run()
        self._thread_active = True
        threading.Thread(target=self.run_process).start()

    def shutdown(self):
        super().shutdown()
        self._thread_active = False

    def process_update(self, update_message):
        self.terminal_logger.info("process_update")
        if not update_message:
            return None

        for iu, ut in update_message:
            if isinstance(iu, GestureIU):
                self.terminal_logger.info("message received from NonverbalGenerator", iu=iu.__dict__)

                # check interruptions
                if self.interrupted_iu is not None:
                    # if, after an interrupted turn, an IU from a new turn has been received
                    if not iu.final and self.interrupted_iu.turn_id != iu.turn_id:
                        self.interrupted_iu = None
                        self.current_input.append(iu)
                elif self.soft_interrupted_iu is not None:
                    self.terminal_logger.info(
                        "IU received during soft interruption",
                        debug=True,
                        soft_inter_iu_turn=self.soft_interrupted_iu.turn_id,
                        TTS_iu_turn=iu.turn_id,
                        iu_final=iu.final,
                    )
                    if hasattr(iu, "final") and iu.final:
                        self.last_clause_each_turn[iu.turnID] = self.last_clause_each_turn_temp[iu.turnID]
                        del self.last_clause_each_turn_temp[iu.turnID]
                        self.terminal_logger.info("DICT updated : ", dict=self.last_clause_each_turn)
                        self.file_logger.info("turn generated")
                        self.interrupted_turn_iu_buffer.append(iu)
                    else:
                        if self.soft_interrupted_iu.turn_id != iu.turn_id:
                            self.soft_interrupted_iu = None
                            self.current_input.append(iu)
                            self.interrupted_turn_iu_buffer = []
                        else:
                            self.interrupted_turn_iu_buffer.append(iu)
                else:
                    self.current_input.append(iu)
                    if hasattr(iu, "final") and iu.final:
                        try:
                            self.last_clause_each_turn[iu.turnID] = self.last_clause_each_turn_temp[iu.turnID]
                            del self.last_clause_each_turn_temp[iu.turnID]
                        except Exception:
                            print(f"last_clause_each_turn_temp do not have a value for key={iu.turnID}")
                        self.terminal_logger.info("DICT updated : ", dict=self.last_clause_each_turn)
                        self.file_logger.info("turn generated")
                    else:
                        self.last_clause_each_turn_temp[iu.turnID] = iu.clauseID

            if isinstance(iu, DMIU):
                if ut == retico_core.UpdateType.ADD:

                    if iu.action == "hard_interruption":
                        self.terminal_logger.info("hard_interruption")
                        self.file_logger.info("hard_interruption")
                        self.first_clause = True

                        # if some iu was outputted, send to LLM module for alignement
                        if self.last_command_started_but_not_ended is not None:
                            output_iu = self.create_speaker_alignement_iu(
                                # grounded_word=self.last_command_started_but_not_ended.grounded_word,
                                # word_id=self.last_command_started_but_not_ended.word_id,
                                # char_id=self.last_command_started_but_not_ended.char_id,
                                clause_id=self.last_command_started_but_not_ended.clause_id,
                                turn_id=self.last_command_started_but_not_ended.turn_id,
                                event="interruption",
                            )
                            um = retico_core.UpdateMessage()
                            um.add_ius(
                                [(um_iu, retico_core.UpdateType.ADD) for um_iu in self.current_output + [output_iu]]
                            )
                            self.append(um)
                            self.interrupted_iu = output_iu
                            # remove all audio in audio_buffer
                            self.current_input = []
                            self.current_output = []
                            # self.last_command_started_but_not_ended = None
                        else:
                            self.terminal_logger.info("speaker interruption but no outputted audio yet")
                            self.file_logger.info("speaker interruption but no outputted audio yet")

                    elif iu.action == "soft_interruption":
                        self.terminal_logger.info(
                            "soft_interruption",
                            debug=True,
                            # grounded_word=self.last_command_started_but_not_ended.grounded_word,
                            # word_id=self.last_command_started_but_not_ended.word_id,
                            # char_id=self.last_command_started_but_not_ended.char_id,
                            clause_id=self.last_command_started_but_not_ended.clause_id,
                            turn_id=self.last_command_started_but_not_ended.turn_id,
                            final=iu.final,
                        )
                        self.file_logger.info("soft_interruption")
                        # if some iu was outputted, send to LLM module for alignement
                        if self.last_command_started_but_not_ended is not None:
                            output_iu = self.create_speaker_alignement_iu(
                                # grounded_word=self.last_command_started_but_not_ended.grounded_word,
                                # word_id=self.last_command_started_but_not_ended.word_id,
                                # char_id=self.last_command_started_but_not_ended.char_id,
                                clause_id=self.last_command_started_but_not_ended.clause_id,
                                turn_id=self.last_command_started_but_not_ended.turn_id,
                                final=iu.final,
                                event="interruption",
                            )
                            um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
                            self.append(um)
                            self.soft_interrupted_iu = output_iu
                            self.interrupted_turn_iu_buffer = self.current_input
                            self.current_input = []

                        else:
                            self.terminal_logger.info("speaker soft interruption but no outputted audio yet")
                            self.file_logger.info("speaker soft interruption but no outputted audio yet")

                    elif iu.action == "stop_turn_id":
                        self.first_clause = True

                    elif iu.action == "continue":
                        self.terminal_logger.info("continue")
                        self.file_logger.info("continue")
                        output_iu = self.create_speaker_alignement_iu(
                            event="continue",
                        )
                        um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
                        self.append(um)
                        self.current_input = self.interrupted_turn_iu_buffer
                        self.soft_interrupted_iu = None

                    elif iu.event == "user_BOT_same_turn":
                        self.interrupted_iu = None

            elif isinstance(iu, UnityMessageIU):

                self.terminal_logger.info(
                    "message received from Unity", dict=self.last_clause_each_turn, iu=iu.__dict__
                )

                if iu.status == "start":
                    self.terminal_logger.info("command started", command=iu.requestID)
                    self.file_logger.info("command started", command=iu.requestID)

                    if self.last_command_started_but_not_ended is None or (
                        self.last_command_started_but_not_ended.turnID is not None
                        and iu.turnID is not None
                        and self.last_command_started_but_not_ended.turnID < iu.turnID
                    ):
                        self.file_logger.info("unity_agent_BOT")
                        output_iu = self.create_speaker_alignement_iu(
                            clause_id=iu.clauseID, turn_id=iu.turnID, event="agent_BOT"
                        )
                        um = retico_core.UpdateMessage()
                        um.add_iu(output_iu, retico_core.UpdateType.ADD)
                        self.append(um)
                    self.last_command_started_but_not_ended = iu

                    # testing with John's space key
                    if iu.requestID[0:5] == "billy" and len(self.last_clause_each_turn) != 0:
                        turn = list(self.last_clause_each_turn.keys())[-1]
                        clause = self.last_clause_each_turn[turn]

                        if self.last_command_started_but_not_ended is None or (
                            self.last_command_started_but_not_ended.turnID is not None
                            and turn is not None
                            and self.last_command_started_but_not_ended.turnID < turn
                        ):
                            self.file_logger.info("unity_agent_BOT")
                            output_iu = self.create_speaker_alignement_iu(
                                clause_id=clause, turn_id=turn, event="agent_BOT"
                            )
                            um = retico_core.UpdateMessage()
                            um.add_iu(output_iu, retico_core.UpdateType.ADD)
                            self.append(um)
                    self.last_command_started_but_not_ended = iu
                elif iu.status == "completed":
                    self.terminal_logger.info("command completed", command=iu.requestID)
                    self.file_logger.info("command completed", command=iu.requestID)
                    self.last_command_ended = iu
                    if self.last_command_started_but_not_ended.requestID == iu.requestID:
                        self.last_command_started_but_not_ended = None
                    # check if EOT
                    if iu.turnID in self.last_clause_each_turn and self.last_clause_each_turn[iu.turnID] == iu.clauseID:
                        self.terminal_logger.info("agent_EOT")
                        self.file_logger.info("unity_EOT")
                        self.send_EOT(iu.turnID, iu.clauseID)

                    # testing with John's space key
                    if iu.requestID[0:5] == "billy" and len(self.last_clause_each_turn) != 0:
                        turn = list(self.last_clause_each_turn.keys())[-1]
                        clause = self.last_clause_each_turn[turn]
                        # create and send IU
                        self.terminal_logger.info(f"EOT : Turn {turn} finished (clause {clause})")
                        # self.terminal_logger.info("agent_EOT")
                        self.file_logger.info("unity_EOT")
                        self.send_EOT(turn, clause)
                elif iu.status == "interrupted":
                    self.terminal_logger.info("command interrupted", command=iu.requestID)
                    self.file_logger.info("command interrupted", command=iu.requestID)
                    self.file_logger.info("unity_interruption")
                    output_iu = self.create_speaker_alignement_iu(
                        clause_id=iu.clauseID, turn_id=iu.turnID, event="interruption"
                    )
                    um = retico_core.UpdateMessage()
                    um.add_iu(output_iu, retico_core.UpdateType.ADD)
                    self.append(um)

                    # testing with John's space key
                    if iu.requestID[0:5] == "billy" and len(self.last_clause_each_turn) != 0:
                        self.terminal_logger.info("command interrupted", command=iu.requestID)
                        self.file_logger.info("command interrupted", command=iu.requestID)
                        self.file_logger.info("unity_interruption")
                        turn = list(self.last_clause_each_turn.keys())[-1]
                        clause = self.last_clause_each_turn[turn]
                        output_iu = self.create_speaker_alignement_iu(
                            clause_id=clause, turn_id=turn, event="interruption"
                        )
                        um = retico_core.UpdateMessage()
                        um.add_iu(output_iu, retico_core.UpdateType.ADD)
                        self.append(um)
                elif iu.status == "aborted":
                    self.terminal_logger.info("command aborted", command=iu.requestID)
                    self.file_logger.info("command aborted", command=iu.requestID)

    def send_EOT(self, turnID, clauseID):
        # clear from dict
        del self.last_clause_each_turn[turnID]

        # create and send IUs
        output_ius = []
        # IU 1
        output_ius.append(
            self.create_speaker_alignement_iu(
                turn_id=turnID,
                clause_id=clauseID,
                event="ius_from_last_turn",
            )
        )
        # IU 2
        output_ius.append(
            self.create_speaker_alignement_iu(
                turn_id=turnID,
                clause_id=clauseID,
                event="agent_EOT",
            )
        )

        um = retico_core.UpdateMessage()
        # um.add_ius([(um_iu, retico_core.UpdateType.ADD) for um_iu in self.current_output + [output_iu]])
        # self.current_output = []
        um.add_ius([(output_iu, retico_core.UpdateType.ADD) for output_iu in output_ius])
        self.append(um)

    def create_speaker_alignement_iu(self, clause_id, turn_id, event, final=True):
        return SpeakerAlignementIU(
            creator=self,
            iuid=f"{hash(self)}:{self.iu_counter}",
            previous_iu=self._previous_iu,
            clause_id=clause_id,
            turn_id=turn_id,
            event=event,
            final=final,
        )

    def run_process(self):
        while self._thread_active:
            if len(self.current_input) == 0:
                time.sleep(0.1)
            else:
                output_iu = self.current_input.pop(0)
                if hasattr(output_iu, "final") and output_iu.final:
                    self.terminal_logger.info("agent_EOT")
                    self.file_logger.info("EOT")
                else:
                    self.terminal_logger.info("EOC")
                    if self.first_clause:
                        self.terminal_logger.info("start_answer_generation")
                        self.file_logger.info("start_answer_generation")
                        self.first_clause = False
                    self.current_turn_id = output_iu.turnID

                um = retico_core.UpdateMessage()
                um.add_iu(output_iu, retico_core.UpdateType.ADD)
                self.append(um)


"""
public class Response { //send result upon either COMMAND received, started, completed, interrupted, aborted
//Diagnostic
public string timestamp = ""; //"12:34:12"; //when send message, request receive by unity
public string requestID = ""; //"152702025787:45544" // send same request ID back?

//Turn/Clause Command IDs
public int turnID = 0; //21345;
public int clauseID = 0; //23154;

public string status = ""; //"start", "completed", "interrupted" , "aborted"

//command time start and end
public string timeStart; //timestamp for when command STARTED
public string timeEnd; //timestamp for when command ENDED (completed or interrupted)

//Index upon with which result was interrupted or completed
public int timingIndex = 0;


public class Audio : Action {
    public float startTime = 0;
    public float endTime = 1;

    public string path = "something.wav";
    public bytes bytes = "/x00/x00";
    public string transcription = "I said this";
    public float volume = 1f;

    public float pitch = 1f;
}
"""
