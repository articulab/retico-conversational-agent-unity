"""
Additional IUs
==============

Additional Incremental Unit classes used in Simple Retico Agent.
"""

import retico_core


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
        interrupt=None,
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
        self.interrupt = interrupt
