import argparse
from functools import partial
import json
import multiprocessing
import os
import pathlib
import signal
import subprocess
import threading
import time
import msgpack
import psutil
import stomp

import retico_core
import retico_amq as amq
from retico_amq.main import callback_fun


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


def find_activemq():
    # need admin to run activemq from ProgramFiles
    # locations_to_check = [
    #     os.environ["ProgramFiles"],
    #     os.environ["ProgramFiles(x86)"],
    # ]
    # locations_to_check = [os.path.join(os.path.expanduser("~"), "Documents")]
    locations_to_check = [
        os.path.join(os.path.expanduser("~"), "Documents"),
        # os.path.join(os.path.expanduser("~"), "Desktop"),
        # os.path.expanduser("~"),
    ]
    for loc in locations_to_check:
        for root, dirs, files in os.walk(loc):
            if "activemq" in files:
                full_path = os.path.join(root, "activemq")
                print(f"Found at: {full_path}")
                return full_path
    else:
        print(f"activemq file not found in {locations_to_check}")


def callback_test(self, update_msg):
    print("callback_test")
    if self.received_iu is None:
        self.received_iu = []
    for iu, ut in update_msg:
        self.received_iu.append(iu)


def thread_network(module: object):
    # running system
    try:
        retico_core.network.run(module)
        print("test running until ENTER key is pressed")
        while True:
            time.sleep(1)
        # retico_core.network.stop(module)
    except Exception:
        print("exception in main")
        retico_core.network.stop(module)


def run_tests():
    # start ActiveMQ subproccess
    process = subprocess.Popen(
        [find_activemq(), "start"],
        shell=True,
    )

    # parameters
    destination = "/topic/retico_out"
    ip = "localhost"
    port = 61613
    is_bytes = True
    # hosts = [("localhost", 61613)]

    # Modules
    cback = retico_core.debug.CallbackModule(callback=callback_fun)
    cback.callback = partial(callback_test, cback)
    cback.received_iu = []
    ar = amq.AMQReader(ip=ip, port=port, message_is_bytes=is_bytes)
    ar.subscribe(cback)

    # proc = threading.Thread(target=thread_network)
    # proc.start()
    proc = multiprocessing.Process(target=thread_network, args=(ar,))
    proc.start()
    proc.join()
    # Terminate the process

    # run tests
    for test in [test_sending_bytes_command]:
        test(destination=destination, callback=cback)

    # kill ActiveMQ subproccess
    # proc.stop()
    proc.terminate()  # sends a SIGTERM
    process.terminate()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "java" in proc.info["name"].lower() and "activemq" in " ".join(proc.info["cmdline"]).lower():
                print(f"Killing PID {proc.pid}")
                proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def test_sending(amq_type: str, data_type: str, destination: str, callback: object):
    # data
    audio_file = f"{pathlib.Path(__file__).parent.resolve()}/audios/hello48k.wav"
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

    if amq_type == "bytes":
        message_is_bytes = True
        headers = {"content-length": 12345}
        if data_type == "audios":
            data = bytes_data
        else:
            data = dict_command
            data["audios"][0]["bytes"] = bytes_data
    else:
        message_is_bytes = False
        headers = {}
        if data_type == "audio":
            raise NotImplementedError("JSON message type not implemented for only audio data")
        else:
            data = dict_command
            data["audios"][0]["path"] = audio_file

    hosts = [("localhost", 61613)]

    send_message(hosts, destination, headers, data, message_is_bytes)

    # wait for message to be received
    time.sleep(5)

    print("Received IU:", callback.received_iu)

    assert len(callback.received_iu) == 1

    iu = callback.received_iu[0]

    assert iu.type == amq.GestureIU
    assert iu.data["audios"][0]["bytes"] == dict_command["audios"][0]["bytes"]
    assert iu.data == dict_command
    # assert iu.data["requestID"] == dict_command["requestID"]
    # assert iu.data["turnID"] == dict_command["turnID"]
    # assert iu.data["clauseID"] == dict_command["clauseID"]
    # assert iu.data["interrupts"] == dict_command["interrupts"]
    # assert iu.data["timings"] == dict_command["timings"]
    # assert iu.data["audios"] == dict_command["audios"]
    # assert iu.data["animations"] == dict_command["animations"]


def test_sending_bytes_command(destination: str, callback: object):
    test_sending(amq_type="bytes", data_type="command", destination=destination, callback=callback)


if __name__ == "__main__":

    run_tests()
