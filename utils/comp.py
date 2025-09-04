import ffmpeg
import sys

if len(sys.argv) != 3:
    print("Usage: python compress.py <input> <output>")
    sys.exit(1)

inp = sys.argv[1]
out = sys.argv[2]

(
    ffmpeg
    .input(inp)
    .output(out,
            vcodec='libx264', crf=28, preset='slow',
            acodec='aac', audio_bitrate='128k')
    .run(overwrite_output=True)
)