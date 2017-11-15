""" Plays back tiny performances by sending OSC messages to Pure Data
    Uses the pyOSC library, so only works under Python 2.7.x """

import struct
import socket
import random
from threading import Timer

DEFAULT_ADDRESS = "localhost"
DEFAULT_PORT = 5000
INT_FLOAT_DGRAM_LEN = 4
STRING_DGRAM_PAD = 4

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setblocking(0)


def send_osc_message(osc_datagram, address, port):
    """Send OSC message via UDP."""
    sock.sendto(osc_datagram, (address, port))


def pad_dgram_four_bytes(dgram):
    """Pad a datagram up to a multiple of 4 bytes."""
    return (dgram + (b'\x00' * (4 - len(dgram) % 4)))


def touch_message_datagram(x, y, z):
    """Constructs an OSC messages with three values to trigger a touch sound."""
    dgram = b''
    dgram += pad_dgram_four_bytes("/touch".encode('utf-8'))
    dgram += pad_dgram_four_bytes(",sfsfsf")
    dgram += pad_dgram_four_bytes("/x".encode('utf-8'))
    dgram += struct.pack('>f', x)
    dgram += pad_dgram_four_bytes("/y".encode('utf-8'))
    dgram += struct.pack('>f', y)
    dgram += pad_dgram_four_bytes("/z".encode('utf-8'))
    dgram += struct.pack('>f', z)
    return dgram


def sound_message_datagram(instrument='strings'):
    dgram = b''
    dgram += pad_dgram_four_bytes("/inst".encode('utf-8'))
    dgram += pad_dgram_four_bytes(",s")
    dgram += pad_dgram_four_bytes(instrument.encode('utf-8'))
    return dgram


# High level sound functions.


def setSynth(instrument="strings"):
    """Sends an OSC message to set the synth instrument."""
    # client.sendto(OSC.OSCMessage("/inst", [instrument]), address)
    synth_dgram = sound_message_datagram(instrument=instrument)
    send_osc_message(synth_dgram, DEFAULT_ADDRESS, DEFAULT_PORT)


def chooseRandomSynth():
    """Choose a random synth for performance playback"""
    setSynth(random.choice(["chirp", "keys", "drums", "strings"]))


def sendTouch(x, y, z):
    """Sends an OSC message to trigger a touch sound."""
    # client.sendto(OSC.OSCMessage("/touch", ["/x", x, "/y", y, "/z", z]), address)
    touch_dgram = touch_message_datagram(x, y, z)
    send_osc_message(touch_dgram, DEFAULT_ADDRESS, DEFAULT_PORT)


def playPerformance(perf):
    """Schedule performance of a tiny performance dataframe."""
    for row in perf.iterrows():
        Timer(row[0], sendTouch, args=[row[1]['x'], row[1]['y'], row[1]['z']]).start()


def playPerformance_XY_only(perf, z=20.0):
    """Schedule playback of a tiny performance dataframe with fake z-values."""
    for row in perf.iterrows():
        # Timer(row[2],sendTouch,args=[row[0]['x'],row[1]['y'],z]).start() # used with time index
        Timer(row[1].time, sendTouch, args=[row[1].x, row[1].y, z]).start()  # used with time in column
