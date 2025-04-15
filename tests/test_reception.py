import argparse

from functools import partial

import retico_core
from retico_core.log_utils import (
    filter_cases,
    configurate_logger,
)

import retico_amq as amq
from retico_amq.main import callback_audio_AMQReader, callback_fun, callback_gesture_AMQReader


def main(
    iu_type: str,
    destination: str,
    is_bytes: bool,
    ip="localhost",
    port=61613,
):
    log_folder = "logs/run"

    # filters
    filters = [
        partial(
            filter_cases,
            cases=[
                # [("debug", [True])],
                [("level", ["warning", "error", "exception"])],
                [
                    (
                        "module",
                        [
                            "AMQReader Module",
                            "AMQWriter Module",
                            "Callback Debug Module",
                            "AudioSaver Module",
                            "TestAudioTurnIUProducing Module",
                        ],
                    )
                ],
            ],
        )
    ]

    # configurate logger
    terminal_logger, _ = configurate_logger(log_folder, filters=filters)

    # Modules
    cback = retico_core.debug.CallbackModule(callback=callback_fun)
    ar = amq.AMQReader(ip=ip, port=port, message_is_bytes=is_bytes)

    # network
    ar.subscribe(cback)

    # additions depending on the tested IU type
    if iu_type == "audio":
        ar.add(destination=destination, target_iu_type=retico_core.audio.AudioIU)
        cback.callback = partial(callback_audio_AMQReader, module=cback)
    elif iu_type == "command":
        ar.add(destination=destination, target_iu_type=amq.GestureIU)
        cback.callback = partial(callback_gesture_AMQReader, module=cback)

    # running system
    try:
        retico_core.network.run(ar)
        print("test running until ENTER key is pressed")
        input()
        retico_core.network.stop(ar)
    except Exception:
        terminal_logger.exception("exception in main")
        retico_core.network.stop(ar)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--amq-type",
        "-a",
        help="Select AMQWriter messages type between bytes and json.",
        type=str,
        default="bytes",
        choices=["bytes", "json"],
    )
    parser.add_argument(
        "--data-type",
        "-d",
        help="Select the type of data that will be sent between audio and command.",
        type=str,
        default="command",
        choices=["audio", "command"],
    )
    parser.add_argument(
        "--destination",
        "-dest",
        help="Select the AMQ topic destination.",
        type=str,
        default="/topic/retico_out",
    )
    args = parser.parse_args()
    print(args)

    if args.amq_type == "bytes":
        message_is_bytes = True
    else:
        message_is_bytes = False

    hosts = ["localhost", 61613]
    destination = args.destination

    main(iu_type=args.data_type, destination=destination, is_bytes=message_is_bytes, ip=hosts[0], port=hosts[1])
