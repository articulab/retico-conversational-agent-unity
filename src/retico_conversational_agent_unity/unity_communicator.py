import retico_core

from retico_conversational_agent_unity.additional_IUs import DMIU, SpeakerAlignementIU
import retico_amq.utils as amqu


class UnityMessageIU(retico_core.abstract.IncrementalUnit):

    @staticmethod
    def type():
        return "Unity Message IU"

    def __init__(
        self,
        timestamp=None,
        requestID=None,
        turnID=None,
        clauseID=None,
        status=None,
        timeStart=None,
        timeEnd=None,
        timingIndex=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.timestamp = timestamp
        self.requestID = requestID
        self.turnID = turnID
        self.clauseID = clauseID
        self.status = status
        self.timeStart = timeStart
        self.timeEnd = timeEnd
        self.timingIndex = timingIndex


class UnityCommunicatorModule(retico_core.abstract.AbstractModule):
    @staticmethod
    def name():
        return "UnityCommunicator Module"

    @staticmethod
    def description():
        return "A module that sends and receives IUs to the Unity side."

    @staticmethod
    def input_ius():
        return [DMIU, UnityMessageIU, amqu.GestureIU]

    @staticmethod
    def output_iu():
        return retico_core.abstract.IncrementalUnit  # SpeakerAlignementIU, amqu.GestureIU

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_clause_each_turn = dict()
        self.latest_processed_iu = None
        self.last_clause_each_turn_temp = dict()

    def process_update(self, update_message):
        self.terminal_logger.info("process_update")
        if not update_message:
            return None

        for iu, ut in update_message:
            if isinstance(iu, amqu.GestureIU):
                self.terminal_logger.info("message received from NonverbalGenerator", iu=iu.__dict__)
                if hasattr(iu, "final") and iu.final:
                    self.last_clause_each_turn[iu.turnID] = self.last_clause_each_turn_temp[iu.turnID]
                    del self.last_clause_each_turn_temp[iu.turnID]
                    self.terminal_logger.info("DICT updated : ", dict=self.last_clause_each_turn)
                    self.file_logger.info("turn generated")
                else:
                    self.last_clause_each_turn_temp[iu.turnID] = iu.clauseID

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

            elif isinstance(iu, UnityMessageIU):

                # # agent EOT
                # if iu.event == "agent_EOT":
                #     self.VA_agent = False
                # if iu.event == "interruption":
                #     self.VA_agent = False
                # # agent BOT
                # elif iu.event == "agent_BOT":
                #     self.VA_agent = True
                # elif iu.event == "continue":
                #     self.VA_agent = True

                self.terminal_logger.info(
                    "message received from Unity", dict=self.last_clause_each_turn, iu=iu.__dict__
                )

                if iu.status == "start":
                    self.terminal_logger.info("command started", command=iu.requestID)
                    self.file_logger.info("command started", command=iu.requestID)
                    if self.latest_processed_iu is None or (
                        self.latest_processed_iu.turnID is not None
                        and iu.turnID is not None
                        and self.latest_processed_iu.turnID < iu.turnID
                    ):
                        self.file_logger.info("agent_BOT")
                        output_iu = self.create_iu(clause_id=iu.clauseID, turn_id=iu.turnID, event="agent_BOT")
                        um = retico_core.UpdateMessage()
                        um.add_iu(output_iu, retico_core.UpdateType.ADD)
                        self.append(um)
                    self.latest_processed_iu = iu
                elif iu.status == "completed":
                    self.terminal_logger.info("command completed", command=iu.requestID)
                    self.file_logger.info("command completed", command=iu.requestID)
                    # check if EOT
                    if iu.turnID in self.last_clause_each_turn and self.last_clause_each_turn[iu.turnID] == iu.clauseID:
                        self.terminal_logger.info("agent_EOT")
                        self.file_logger.info("EOT")
                        self.send_EOT(iu.turnID, iu.clauseID)

                    # testing with John's space key
                    if iu.requestID[0:5] == "billy" and len(self.last_clause_each_turn) != 0:
                        turn = list(self.last_clause_each_turn.keys())[-1]
                        clause = self.last_clause_each_turn[turn]
                        # create and send IU
                        self.terminal_logger.info(f"EOT : Turn {turn} finished (clause {clause})")
                        # self.terminal_logger.info("agent_EOT")
                        self.file_logger.info("EOT")
                        self.send_EOT(turn, clause)
                elif iu.status == "interrupted":
                    self.terminal_logger.info("command aborted", command=iu.requestID)
                    self.file_logger.info("command aborted", command=iu.requestID)
                    self.file_logger.info("interruption")
                    output_iu = self.create_iu(clause_id=iu.clauseID, turn_id=iu.turnID, event="interruption")
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
            self.create_iu(
                turn_id=turnID,
                # clause_id=clauseID,
                final=False,
                event="ius_from_last_turn",
            )
        )
        # IU 2
        output_ius.append(
            self.create_iu(
                turn_id=turnID,
                # clause_id=clauseID,
                final=True,
                event="agent_EOT",
            )
        )

        um = retico_core.UpdateMessage()
        # um.add_ius([(um_iu, retico_core.UpdateType.ADD) for um_iu in self.current_output + [output_iu]])
        # self.current_output = []
        um.add_ius([(output_iu, retico_core.UpdateType.ADD) for output_iu in output_ius])
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
    public string transcription = "I said this";
    public float volume = 1f;

    public float pitch = 1f;
}
"""
