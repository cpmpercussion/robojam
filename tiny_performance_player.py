""" Plays back tiny performances by sending OSC messages to Pure Data
    Uses the pyOSC library, so only works under Python 2.7.x """

import OSC
import random
from threading import Timer

client = OSC.OSCClient()
address = ("localhost", 5000)


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
