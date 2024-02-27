'''updated the function mmap_read'''
def mmap_read(path: str, offset: int, length: int) -> bytes:
    output_byteio = io.BytesIO()
    audio, sr = soundfile.read(path, start=offset, stop=offset + length)
    soundfile.write(output_byteio, audio, samplerate=sr, format='wav')
    return output_byteio.getvalue()