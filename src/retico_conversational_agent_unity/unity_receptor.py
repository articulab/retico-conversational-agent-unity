import retico_core

from retico_conversational_agent_unity.additional_IUs import SpeakerAlignementIU
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


class UnityReceptorModule(retico_core.abstract.AbstractModule):
    @staticmethod
    def name():
        return "UnityReceptor Module"

    @staticmethod
    def description():
        return "A module that receives and process Unity messages."

    @staticmethod
    def input_ius():
        return [UnityMessageIU, amqu.GestureIU]

    @staticmethod
    def output_iu():
        return SpeakerAlignementIU

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_clause_each_turn = dict()

    def process_update(self, update_message):
        self.terminal_logger.info("process_update")
        if not update_message:
            return None

        for iu, ut in update_message:
            if isinstance(iu, amqu.GestureIU):
                # self.terminal_logger.info("message received from GestureProducer", iu=iu.__dict__)
                if hasattr(iu, "final") and iu.final:
                    self.last_clause_each_turn[iu.turnID] = iu.clauseID
                    self.terminal_logger.info("DICT updated : ", dict=self.last_clause_each_turn)

            elif isinstance(iu, UnityMessageIU):
                self.terminal_logger.info(
                    "message received from Unity", dict=self.last_clause_each_turn, iu=iu.__dict__
                )

                # check if turn completed
                if iu.turnID in self.last_clause_each_turn and self.last_clause_each_turn[iu.turnID] == iu.clauseID:
                    # clear from dict
                    del self.last_clause_each_turn[iu.turnID]

                    # create and send IU
                    self.terminal_logger.info("agent_EOT")
                    self.file_logger.info("EOT")
                    output_iu = self.create_iu(
                        grounded_word=None,
                        word_id=None,
                        char_id=None,
                        clause_id=iu.clauseID,
                        turn_id=iu.turnID,
                        final=True,
                        event="agent_EOT",
                    )

                    um = retico_core.UpdateMessage()
                    # um.add_ius([(um_iu, retico_core.UpdateType.ADD) for um_iu in self.current_output + [output_iu]])
                    # self.current_output = []
                    um.add_iu(output_iu, retico_core.UpdateType.ADD)
                    self.append(um)

                # testing wiith John's space key
                if iu.requestID[0:5] == "billy" and len(self.last_clause_each_turn) != 0:
                    turn = list(self.last_clause_each_turn.keys())[-1]
                    clause = self.last_clause_each_turn[turn]

                    # clear from dict
                    del self.last_clause_each_turn[turn]

                    # create and send IU
                    self.terminal_logger.info("agent_EOT")
                    self.file_logger.info("EOT")
                    output_iu = self.create_iu(
                        grounded_word=None,
                        word_id=None,
                        char_id=None,
                        clause_id=clause,
                        turn_id=turn,
                        final=True,
                        event="agent_EOT",
                    )

                    um = retico_core.UpdateMessage()
                    # um.add_ius([(um_iu, retico_core.UpdateType.ADD) for um_iu in self.current_output + [output_iu]])
                    # self.current_output = []
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
    public string transcription = "I said this";
    public float volume = 1f;

    public float pitch = 1f;
}
"""
