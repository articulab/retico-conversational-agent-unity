"""
Additional IUs
==============

Additional Incremental Unit classes used in Simple Retico Agent.
"""

import retico_core

class SpeechRecognitionTurnIU(retico_core.text.SpeechRecognitionIU):
    """Same IU as SpeechRecognition, but enhanced with turn_id."""

    @staticmethod
    def type():
        return "SpeechRecognitionTurn IU"

    def __init__(self, turn_id=None, **kwargs):
        super().__init__(**kwargs)
        self.turn_id = turn_id

class TextFinalIU(retico_core.text.TextIU):
    """TextIU with an additional final attribute."""

    @staticmethod
    def type():
        return "Text Final IU"

    def __init__(self, final=False, **kwargs):
        super().__init__(
            **kwargs,
        )
        self.final = final


class AudioFinalIU(retico_core.audio.AudioIU):
    """AudioIU with an additional final attribute."""

    @staticmethod
    def type():
        return "Audio Final IU"

    def __init__(self, final=False, **kwargs):
        super().__init__(
            **kwargs,
        )
        self.final = final


class VADIU(retico_core.audio.AudioIU):
    """AudioIU enhanced by VADModule with VA for both user and agent.

    Attributes:
        va_user (bool): user VA activation, True means voice recognized,
            False means no voice recognized.
        va_agent (bool): agent VA activation, True means audio outputted
            by the agent, False means no audio outputted by the agent.
    """

    @staticmethod
    def type():
        return "VAD IU"

    def __init__(
        self,
        va_user=None,
        va_agent=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.va_user = va_user
        self.va_agent = va_agent


class BackchannelIU(retico_core.audio.AudioIU):
    """AudioIU with a different type for backchannel behavior."""

    @staticmethod
    def type():
        return "Backchannel IU"

    def __init__(self, **kwargs):
        super().__init__(
            **kwargs,
        )


class DMIU(retico_core.audio.AudioIU):

    @staticmethod
    def type():
        return "DM IU"

    def __init__(
        self,
        action=None,
        event=None,
        turn_id=None,
        word_id=None,
        char_id=None,
        clause_id=None,
        grounded_word=None,
        final=None,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.action = action
        self.event = event
        self.turn_id = turn_id
        self.word_id = word_id
        self.char_id = char_id
        self.clause_id = clause_id
        self.grounded_word = grounded_word
        self.final = final


class SpeakerAlignementIU(retico_core.audio.AudioIU):
    """AudioIU enhanced with information that aligns the AudioIU to the current
    written agent turn.

    Attributes:
        grounded_word (str): the word corresponding to the audio.
        turn_id (int): The index of the dialogue's turn the IU is part
            of.
        clause_id (int): The index of the clause the IU is part of, in
            the current turn.
        word_id (int): The index of the word that corresponds to the end
            of the IU].
        char_id (int): The index of the last character from the
            grounded_word.
        final (bool): Wether the IU is an EOT.
    """

    @staticmethod
    def type():
        return "SpeakerAlignement IU"

    def __init__(
        self,
        creator=None,
        iuid=0,
        previous_iu=None,
        grounded_in=None,
        grounded_word=None,
        word_id=None,
        char_id=None,
        turn_id=None,
        clause_id=None,
        event=None,
        final=None,
        **kwargs,
    ):
        super().__init__(
            creator=creator,
            iuid=iuid,
            previous_iu=previous_iu,
            grounded_in=grounded_in,
            **kwargs,
        )
        self.grounded_word = grounded_word
        self.word_id = word_id
        self.char_id = char_id
        self.turn_id = turn_id
        self.clause_id = clause_id
        self.event = event
        self.final = final


class TextAlignedAudioIU(retico_core.audio.AudioIU):
    """AudioIU enhanced with information that aligns the AudioIU to the current
    written agent turn.

    Attributes:
        grounded_word (str): the word corresponding to the audio.
        turn_id (int): The index of the dialogue's turn the IU is part
            of.
        clause_id (int): The index of the clause the IU is part of, in
            the current turn.
        word_id (int): The index of the word that corresponds to the end
            of the IU].
        char_id (int): The index of the last character from the
            grounded_word.
        final (bool): Wether the IU is an EOT.
    """

    @staticmethod
    def type():
        return "Text Aligned Audio IU"

    def __init__(
        self,
        creator=None,
        iuid=0,
        previous_iu=None,
        grounded_in=None,
        audio=None,
        rate=None,
        nframes=None,
        sample_width=None,
        grounded_word=None,
        word_id=None,
        char_id=None,
        turn_id=None,
        clause_id=None,
        final=None,
        **kwargs,
    ):
        super().__init__(
            creator=creator,
            iuid=iuid,
            previous_iu=previous_iu,
            grounded_in=grounded_in,
            payload=audio,
            raw_audio=audio,
            rate=rate,
            nframes=nframes,
            sample_width=sample_width,
        )
        self.grounded_word = grounded_word
        self.word_id = word_id
        self.char_id = char_id
        self.turn_id = turn_id
        self.clause_id = clause_id
        self.final = final

    def set_data(
        self,
        grounded_word=None,
        word_id=None,
        char_id=None,
        turn_id=None,
        clause_id=None,
        audio=None,
        chunk_size=None,
        rate=None,
        sample_width=None,
        final=False,
    ):
        """Sets AudioIU parameters and the alignment information."""
        # alignment information
        self.grounded_word = grounded_word
        self.word_id = word_id
        self.char_id = char_id
        self.turn_id = turn_id
        self.clause_id = clause_id
        self.final = final
        # AudioIU information
        self.payload = audio
        self.raw_audio = audio
        self.rate = rate
        self.nframes = chunk_size
        self.sample_width = sample_width


class TurnTextIU(retico_core.text.TextIU):
    """TextIU enhanced with information related to dialogue turns, clauses,
    etc.

    Attributes:
        turn_id (int): Which dialogue's turn the IU is part of.
        clause_id (int): Which clause the IU is part of, in the current
            turn.
        final (bool): Wether the IU is an EOT.
    """

    @staticmethod
    def type():
        return "Turn Text IU"

    def __init__(
        self,
        creator=None,
        iuid=0,
        previous_iu=None,
        grounded_in=None,
        text=None,
        turn_id=None,
        clause_id=None,
        final=False,
        **kwargs,
    ):
        super().__init__(
            creator=creator,
            iuid=iuid,
            previous_iu=previous_iu,
            grounded_in=grounded_in,
            text=text,
        )
        self.turn_id = turn_id
        self.clause_id = clause_id
        self.final = final

    def set_data(
        self,
        text=None,
        turn_id=None,
        clause_id=None,
        final=False,
    ):
        """Sets TextIU parameters and dialogue turns informations (turn_id,
        clause_id, final)"""
        # dialogue turns information
        self.turn_id = turn_id
        self.clause_id = clause_id
        self.final = final
        # TextIU information
        self.payload = text
        self.text = text


class VADTurnAudioIU(retico_core.audio.AudioIU):
    """AudioIU enhanced by VADTurnModule with dialogue turn information
    (agent_turn, user_turn, silence, interruption, etc) contained in the
    vad_state parameter.

    Attributes:
        vad_state (string): dialogue turn information (agent_turn,
            user_turn, silence, interruption, etc) from VADTurnModule.
    """

    @staticmethod
    def type():
        return "VADTurn Audio IU"

    def __init__(
        self,
        creator=None,
        iuid=0,
        previous_iu=None,
        grounded_in=None,
        audio=None,
        vad_state=None,
        rate=None,
        nframes=None,
        sample_width=None,
        **kwargs,
    ):
        super().__init__(
            creator=creator,
            iuid=iuid,
            previous_iu=previous_iu,
            grounded_in=grounded_in,
            payload=audio,
            raw_audio=audio,
            rate=rate,
            nframes=nframes,
            sample_width=sample_width,
        )
        self.vad_state = vad_state

    def set_data(
        self, vad_state=None, audio=None, nframes=None, rate=None, sample_width=None
    ):
        """Sets AudioIU parameters and vad_state."""
        # vad_state
        self.vad_state = vad_state
        # AudioIU information
        self.payload = audio
        self.raw_audio = audio
        self.rate = rate
        self.nframes = nframes
        self.sample_width = sample_width
