""" Plays back tiny performances by sending OSC messages to Pure Data
    Uses the pyOSC library, so only works under Python 2.7.x """

import socket
import random
from threading import Timer

ADDRESS = "localhost"
PORT = 5000
INT_FLOAT_DGRAM_LEN = 4
STRING_DGRAM_PAD = 4

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setblocking(0)

# OSC functions

def send_sound_command(osc_datagram):
    """Send OSC message via UDP."""
    sock.sendto(osc_datagram, (ADDRESS, PORT))

def pad_dgram_four_bytes(dgram):
    """Pad a datagram up to a multiple of 4 bytes."""
    return (dgram + (b'\x00' * (4-len(dgram)%4)))

def touch_message_datagram(pos = 0.0):
    """Construct an osc message with address /touch and one float."""
    dgram = b''
    dgram += pad_dgram_four_bytes("/touch".encode('utf-8'))
    dgram += pad_dgram_four_bytes(",f")
    dgram += struct.pack('>f', pos)
    return dgram

# High level sound functions.

def setSynth(instrument="strings"):
    """Sends an OSC message to set the synth instrument."""
    client.sendto(OSC.OSCMessage("/inst", [instrument]), address)


def chooseRandomSynth():
    """Choose a random synth for performance playback"""
    setSynth(random.choice(["chirp", "keys", "drums", "strings"]))


def sendTouch(x, y, z):
    """Sends an OSC message to trigger a touch sound."""
    client.sendto(OSC.OSCMessage("/touch", ["/x", x, "/y", y, "/z", z]), address)


def playPerformance(perf):
    """Schedule performance of a tiny performance dataframe."""
    for row in perf.iterrows():
        Timer(row[0], sendTouch, args=[row[1]['x'], row[1]['y'], row[1]['z']]).start()


def playPerformance_XY_only(perf, z=20.0):
    """Schedule playback of a tiny performance dataframe with fake z-values."""
    for row in perf.iterrows():
        # Timer(row[2],sendTouch,args=[row[0]['x'],row[1]['y'],z]).start() # used with time index
        Timer(row[1].time, sendTouch, args=[row[1].x, row[1].y, z]).start()  # used with time in column

## Todo, remove dependency on OSC library.