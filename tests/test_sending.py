import argparse
import json
import pathlib
import msgpack
import stomp

import retico_core


def send_message(hosts, destination, headers, data, message_is_bytes):
    try:
        conn = stomp.Connection(
            host_and_ports=hosts,
            auto_content_length=message_is_bytes,
        )
        conn.connect("admin", "admin", wait=True)
    except stomp.exception.ConnectFailedException as e:
        print("Connection failed:", e)

    body = None
    if message_is_bytes:
        try:
            body = msgpack.packb(data)  # Convert dict to binary
        except Exception as e:
            print("Error while packing data as BYTES", e)
    else:
        try:
            body = json.dumps(data, indent=2)
        except Exception as e:
            print("Error while packing data as stringified JSON", e)

    conn.send(
        body=body,
        destination=destination,
        headers=headers,
        persistent=True,
    )


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

    # data
    audio_file = f"{pathlib.Path(__file__).parent.resolve()}/hello48k.wav"
    bytes_data, metadata = retico_core.audio.load_audiofile_WAVPCM16(audio_file)

    # data command
    dict_command = {
        "requestID": 1234345,
        "turnID": 0,
        "clauseID": 0,
        "interrupts": 0,
        "timings": [0],
        "audios": [{"transcription": "Hello,", "volume": 1, "delay": 0.5, "timingIndex": 0}],
        "animations": [{"animation": "standing_greeting", "bodypart": "rightarm", "duration": 2.0, "delay": 0.0}],
    }

    if args.amq_type == "bytes":
        message_is_bytes = True
        headers = {"content-length": 12345}
        if args.data_type == "audios":
            data = bytes_data
        else:
            data = dict_command
            data["audios"][0]["bytes"] = bytes_data
    else:
        message_is_bytes = False
        headers = {}
        if args.data_type == "audio":
            raise NotImplementedError("JSON message type not implemented for only audio data")
        else:
            data = dict_command
            data["audios"][0]["path"] = audio_file

    hosts = [("localhost", 61613)]
    destination = args.destination

    send_message(hosts, destination, headers, data, message_is_bytes)
