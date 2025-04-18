import argparse
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from functools import partial
import torch

import retico_core
from retico_core import network
from retico_core.log_utils import (
    filter_cases,
    configurate_plot,
    plot_once,
)
from retico_wozmic import WOZMicrophoneModule, WOZMicrophoneModule_2
import retico_amq as amq
import retico_conversational_agent as agent
import retico_conversational_agent_unity as uagent


def main_DM_unity():
    """The `main_DM` function creates and runs a dialog system that is able to
    have a conversation with the user.

    The dialog system is composed of different modules: - a Microphone :
    captures the user's voice - an ASR : transcribes the user's voice
    into text - a LLM : generates a textual answer to the trancription
    from user's spoken sentence. - a TTS : generates a spoken answer
    from the LLM's textual answer. - a Speaker : outputs the spoken
    answer generated by the system.

    We provide the system with a scenario (contained in the
    "system_prompt") that it will follow through the conversation : The
    system is a teacher and it will teach mathematics to a 8-year-old
    child student (the user)

    the parameters defined : - model_path : the path to the weights of
    the LLM that will be used in the dialog system. - system_prompt : a
    part of the prompt that will be given to the LLM at every agent turn
    to set the scenario of the conversation. - printing : an argument
    that set to True will print a lot of information useful for
    degugging. - rate : the target audio signal rate to which the audio
    captured by the microphone will be converted to (so that it is
    suitable for every module) - frame_length : the chosen frame length
    in seconds at which the audio signal will be chunked. - log_folder :
    the path to the folder where the logs (information about each
    module's latency) will be saved.

    It is recommended to not modify the rate and frame_length parameters
    because the modules were coded with theses values and it is not
    ensured that the system will run correctly with other values.
    """

    # parameters definition
    device = "cuda" if torch.cuda.is_available() else "cpu"
    printing = False
    log_folder = "logs/run"
    frame_length = 0.02
    tts_frame_length = 0.2
    # tts_frame_length = 0.02
    rate = 16000
    # tts_model_samplerate = 22050
    # tts_model = "vits_vctk"
    tts_model_samplerate = 48000
    tts_model = "jenny"
    model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"
    system_prompt = "This is a spoken dialog scenario between a teacher and a 8 years old child student.\
        The teacher is teaching mathemathics to the child student.\
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        You play the role of a teacher. Here is the beginning of the conversation :"
    plot_config_path = "configs/plot_config_DM.json"
    plot_live = True
    prompt_format_config = "configs/prompt_format_config.json"
    context_size = 2000
    store_audio = False
    ip = "localhost"
    port = "61613"

    # AMQ parameters
    destination_retico_out = "/topic/retico_out"
    destination_unity_out = "/topic/unity_out"

    # filters
    filters = [
        partial(
            filter_cases,
            cases=[
                # [("debug", [True])],
                [
                    (
                        "module",
                        [
                            "NonverbalGenerator Module",
                            "UnityCommunicator Module",
                            "TTS DM Module",
                            "AMQReader Module",
                            "AMQWriter Module",
                        ],
                    )
                ],
                # [("module", ["GestureDemo Module", "UnityReceptor Module"])],
                # [("debug", [True]), ("module", ["LLM DM Module", "TTS DM Module"])],
                [("level", ["warning", "error"])],
            ],
            # cases=[
            #     [("module", ["DialogueManager Module"])],
            #     [("level", ["warning", "error"])],
            # ],
        )
    ]
    # configurate logger
    terminal_logger, _ = retico_core.log_utils.configurate_logger(log_folder, filters=filters)

    # configure plot
    configurate_plot(
        is_plot_live=plot_live,
        refreshing_time=1,
        plot_config_path=plot_config_path,
        window_duration=30,
    )

    dialogue_history = agent.DialogueHistory(
        prompt_format_config,
        terminal_logger=terminal_logger,
        initial_system_prompt=system_prompt,
        context_size=context_size,
    )

    # create modules
    # mic = MicrophonePTTModule(rate=rate, frame_length=frame_length)
    # mic = audio.MicrophoneModule(rate=rate, frame_length=frame_length)
    # mic = audio.MicrophoneModule()
    mic = WOZMicrophoneModule(frame_length=frame_length)

    vad = agent.VadModule(
        input_framerate=rate,
        frame_length=frame_length,
    )

    dm = agent.DialogueManagerModule(
        dialogue_history=dialogue_history,
        input_framerate=rate,
        frame_length=frame_length,
    )
    dm.add_repeat_policy()
    dm.add_soft_interruption_policy()
    dm.add_continue_policy()
    # dm.add_backchannel_policy()

    asr = agent.AsrDmModule(
        device=device,
        full_sentences=True,
        input_framerate=rate,
    )

    llm = agent.LlmDmModule(
        model_path,
        None,
        None,
        dialogue_history=dialogue_history,
        printing=printing,
        device=device,
        # verbose=True,
    )

    tts = agent.TtsDmModule(
        language="en",
        model=tts_model,
        printing=printing,
        frame_duration=tts_frame_length,
        device=device,
        # device="cpu",
    )

    # create network
    mic.subscribe(vad)
    vad.subscribe(dm)
    dm.subscribe(asr)
    dm.subscribe(llm)
    dm.subscribe(tts)
    asr.subscribe(llm)
    llm.subscribe(tts)

    unity_comm = uagent.UnityCommunicatorModule()
    nvg = uagent.NonverbalGeneratorModule(tts_framerate=tts_model_samplerate, store_audio=store_audio)
    # gesture_demo = uagent.GestureDemoModule()
    # gesture_prod_demo = uagent.GestureProducerDemoModule()
    tts.subscribe(nvg)
    nvg.subscribe(unity_comm)
    dm.subscribe(unity_comm)
    unity_comm.subscribe(llm)

    dict_out = [
        {"module": unity_comm, "destination": destination_retico_out},
    ]
    dict_in = [
        {"destination": destination_unity_out, "iu_type": uagent.UnityMessageIU, "subscriber_modules": [unity_comm]},
    ]
    amq.define_amq_network(
        modules_out_dict=dict_out,
        modules_in_dict=dict_in,
        verbose=True,
        ip=ip,
        port=port,
        # message_is_bytes=not store_audio,
        message_out_is_bytes=not store_audio,
        message_in_is_bytes=not store_audio,
    )

    # running system
    try:
        network.run(mic)
        # terminal_logger.info("Dialog system running until ENTER key is pressed")
        print("Dialog system running until ENTER key is pressed")
        input()
        network.stop(mic)
    except Exception:
        terminal_logger.exception("exception in main")
        network.stop(mic)
    finally:
        plot_once(
            plot_config_path=plot_config_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("simple_example")
    parser.add_argument(
        "--cuda_test", "-ct", nargs="+", help="if set, execute cuda_test instead of regular system execution.", type=str
    )
    args = parser.parse_args()
    if args.cuda_test is not None:
        agent.test_cuda(args.cuda_test)
    else:
        main_DM_unity()
    plot_once(plot_config_path="configs/plot_config_DM.json")
