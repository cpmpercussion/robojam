""" Plays back tiny performances by sending OSC messages to Pure Data """
import struct
import socket
import random
from threading import Timer

DEFAULT_OSC_ADDRESS = "localhost"
DEFAULT_OSC_PORT = 5000


class TouchScreenOscClient(object):
    """A simple OSC client for sending messages recording touch screen performances."""

    def __init__(self):
        # just set up the socket.
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(0)

    def send_osc_message(self, osc_datagram, address, port):
        """Send OSC message via UDP."""
        self.sock.sendto(osc_datagram, (address, port))

    def pad_dgram_four_bytes(self, dgram):
        """Pad a datagram up to a multiple of 4 bytes."""
        return (dgram + (b'\x00' * (4 - len(dgram) % 4)))

    def setSynth(self, instrument="strings", address=DEFAULT_OSC_ADDRESS, port=DEFAULT_OSC_PORT):
        """Sends an OSC message to set the synth instrument."""
        dgram = b''
        dgram += self.pad_dgram_four_bytes("/inst".encode('utf-8'))
        dgram += self.pad_dgram_four_bytes(",s")
        dgram += self.pad_dgram_four_bytes(instrument.encode('utf-8'))
        self.send_osc_message(dgram, address, port)

    def setSynthRandom(self):
        """Choose a random synth for performance playback"""
        self.setSynth(random.choice(["chirp", "keys", "drums", "strings"]))

    def sendTouch(self, x, y, z, address=DEFAULT_OSC_ADDRESS, port=DEFAULT_OSC_PORT):
        """Sends an OSC message to trigger a touch sound."""
        dgram = b''
        dgram += self.pad_dgram_four_bytes("/touch".encode('utf-8'))
        dgram += self.pad_dgram_four_bytes(",sfsfsf")
        dgram += self.pad_dgram_four_bytes("/x".encode('utf-8'))
        dgram += struct.pack('>f', x)
        dgram += self.pad_dgram_four_bytes("/y".encode('utf-8'))
        dgram += struct.pack('>f', y)
        dgram += self.pad_dgram_four_bytes("/z".encode('utf-8'))
        dgram += struct.pack('>f', z)
        self.send_osc_message(dgram, address, port)

    def playPerformance(self, perf_df):
        """Schedule performance of a tiny performance dataframe."""
        # Dataframe must have abolute time (in seconds) as index, and 'x', 'y', and 'z' as column names.
        for row in perf_df.iterrows():
            Timer(row[0], self.sendTouch, args=[row[1].x, row[1].y, row[1].z]).start()  # used with time in column
