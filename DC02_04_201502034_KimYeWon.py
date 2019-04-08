from __future__ import print_function

import sys
import wave

from io import StringIO

import alsaaudio
import colorama
import numpy as np
import pyaudio as pya

from reedsolo import RSCodec, ReedSolomonError
from termcolor import cprint
from pyfiglet import figlet_format

HANDSHAKE_START_HZ = 4096
HANDSHAKE_END_HZ = 5120 + 1024

START_HZ = 1024
STEP_HZ = 256
BITS = 4

MY_STD_NUM = "201502034"
MY_STD_LEN = len(MY_STD_NUM)

FEC_BYTES = 4

def stereo_to_mono(input_file, output_file):
    inp = wave.open(input_file, 'r')
    params = list(inp.getparams())
    params[0] = 1 # nchannels
    params[3] = 0 # nframes

    out = wave.open(output_file, 'w')
    out.setparams(tuple(params))

    frame_rate = inp.getframerate()
    frames = inp.readframes(inp.getnframes())
    data = np.fromstring(frames, dtype=np.int16)
    left = data[0::2]
    out.writeframes(left.tostring())

    inp.close()
    out.close()

def yield_chunks(input_file, interval):
    wav = wave.open(input_file)
    frame_rate = wav.getframerate()

    chunk_size = int(round(frame_rate * interval))
    total_size = wav.getnframes()

    while True:
        chunk = wav.readframes(chunk_size)
        if len(chunk) == 0:
            return

        yield frame_rate, np.fromstring(chunk, dtype=np.int16)

def dominant(frame_rate, chunk):
    w = np.fft.fft(chunk)
    freqs = np.fft.fftfreq(len(chunk))
    #print("TEST-len(chunk) : ", len(chunk)) #2205
    peak_coeff = np.argmax(np.abs(w))
    #print("TEST-phase : ",np.argmax(np.angle(w)))
    peak_freq = freqs[peak_coeff] #frequency
     #print(abs(peak_freq * frame_rate))
    return abs(peak_freq * frame_rate) # in Hz

def match(freq1, freq2):
    return abs(freq1 - freq2) < 20

def decode_bitchunks(chunk_bits, chunks):
    out_bytes = []

    next_read_chunk = 0
    next_read_bit = 0

    byte = 0
    bits_left = 8
    while next_read_chunk < len(chunks):
        can_fill = chunk_bits - next_read_bit
        to_fill = min(bits_left, can_fill)
        offset = chunk_bits - next_read_bit - to_fill
        byte <<= to_fill
        shifted = chunks[next_read_chunk] & (((1 << to_fill) - 1) << offset)
        byte |= shifted >> offset;
        bits_left -= to_fill
        next_read_bit += to_fill
        if bits_left <= 0:

            out_bytes.append(byte)
            #print(out_bytes)
            byte = 0
            bits_left = 8

        if next_read_bit >= chunk_bits:
            next_read_chunk += 1
            next_read_bit -= chunk_bits

    return out_bytes

def decode_file(input_file, speed):
    wav = wave.open(input_file)
    if wav.getnchannels() == 2:
        mono = StringIO()
        stereo_to_mono(input_file, mono)

        mono.seek(0)
        input_file = mono
    wav.close()

    offset = 0
    for frame_rate, chunk in yield_chunks(input_file, speed / 2):
        dom = dominant(frame_rate, chunk)
        print("{} => {}".format(offset, dom))
        offset += 1

def extract_packet(freqs):
    freqs = freqs[::2]
    bit_chunks = [int(round((f - START_HZ) / STEP_HZ)) for f in freqs]
    bit_chunks = [c for c in bit_chunks[1:] if 0 <= c < (2 ** BITS)]
    #print(bit_chunks)
    return bytearray(decode_bitchunks(BITS, bit_chunks))

def display(s):
    cprint(figlet_format(s.replace(' ', '   '), font='doom'), 'yellow')

def encode_bitchunks(chunk_bits, decoded_chunks):
    output_bit_array = []
    for dbyte in decoded_chunks:
        temp = []
        bitlist = range(0, 8, chunk_bits)
        offset = 2**chunk_bits - 1
        for i in bitlist:
            temp.append((dbyte>>i) & offset)
        temp.reverse()
        for j in temp:
            output_bit_array.append(j)
    print("output_bit_array : ",output_bit_array)
    return output_bit_array

def output_frequency(encoded_byte_stream, frame_rate):
    #extract_packet() function <-> output_frequency() function
    frequency = [ (element * STEP_HZ + START_HZ) for element in encoded_byte_stream]
    frequency.insert(0,HANDSHAKE_START_HZ)
    frequency.append(HANDSHAKE_END_HZ)
    return frequency

def send_signal(byte_stream, frame_rate=44100):
    p = pya.PyAudio()
    stream = p.open(format=pya.paFloat32, channels=1, rate=frame_rate, output = True)
    encoded_byte_stream = encode_bitchunks(BITS,byte_stream)
    freq_list = output_frequency(encoded_byte_stream, frame_rate)
    samples_list = []
    SETA = 0
    AMPLITUDE = 1
    DURATION = 0.5
    for freq in freq_list:
        sample = (AMPLITUDE * (np.sin(2* np.pi * freq * np.arange(np.ceil(frame_rate * DURATION))/frame_rate + SETA))).astype(np.float32)
        samples_list.append(sample)
    for samples in samples_list:
        stream.write(samples)
    #handshake end~
    #stop stream
    stream.stop_stream()
    stream.close()
    #close Pyaudio
    p.terminate()



def listen_linux(frame_rate=44100, interval=0.1):

    mic = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device="default")
    mic.setchannels(1)
    mic.setrate(44100)
    mic.setformat(alsaaudio.PCM_FORMAT_S16_LE)

    num_frames = int(round((interval / 2) * frame_rate))
     #print("TEST-num_frames : ", num_frames)
    mic.setperiodsize(num_frames)
    print("start...")

    in_packet = False
    packet = []

    while True:
        l, data = mic.read()
        if not l:
            continue

        chunk = np.fromstring(data, dtype=np.int16)
        dom = dominant(frame_rate, chunk)

        if in_packet and match(dom, HANDSHAKE_END_HZ):
            #mic.pause(True)
            byte_stream = extract_packet(packet)
            try:
                byte_stream = RSCodec(FEC_BYTES).decode(byte_stream)
                byte_stream = byte_stream.decode("utf-8")
                if MY_STD_NUM in byte_stream:
                    byte_stream = byte_stream.replace(MY_STD_NUM, "")
                    display(byte_stream)

                    byte_stream = RSCodec(FEC_BYTES).encode(byte_stream)
                    #mic.pause(True)
                    send_signal(byte_stream)
                    #mic.pause(False)
            except ReedSolomonError as e:
                pass
                #print("{}: {}".format(e, byte_stream))

            packet = []
            in_packet = False
        elif in_packet:
            packet.append(dom)
        elif match(dom, HANDSHAKE_START_HZ):
            in_packet = True

if __name__ == '__main__':
    colorama.init(strip=not sys.stdout.isatty())

#decode_file(sys.argv[1], float(sys.argv[2]))
    listen_linux()
