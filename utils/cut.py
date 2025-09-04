#!/usr/bin/env python3
import argparse
import subprocess
import sys

def cut_segment(input_file, start, end, output_file):
    # Erstellt ein Segment mit ffmpeg, kopiert dabei Video und Audio ohne Neukodierung
    command = [
        "ffmpeg",
        "-y",              # Überschreiben ohne Nachfrage
        "-ss", start,      # Startzeit
        "-i", input_file,  # Eingabedatei
        "-to", end,        # Endzeit
        "-c", "copy",    # Stream Copy (kein Encoding)
        output_file         # Ausgabedatei
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Segment erfolgreich erstellt: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Erstellen von {output_file}: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Schneide ein Fußballspiel-Video in zwei Halbzeiten"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Pfad zum Eingabevideo"
    )
    parser.add_argument(
        "--fh-start", required=True,
        help="Startzeit erste Halbzeit im Format HH:MM:SS"
    )
    parser.add_argument(
        "--fh-end", required=True,
        help="Endzeit erste Halbzeit im Format HH:MM:SS"
    )
    """ parser.add_argument(
        "--sh-start", required=True,
        help="Startzeit zweite Halbzeit im Format HH:MM:SS"
    )
    parser.add_argument(
        "--sh-end", required=True,
        help="Endzeit zweite Halbzeit im Format HH:MM:SS"
    ) """
    parser.add_argument(
        "--out1", default="half1.mp4",
        help="Ausgabedatei für erste Halbzeit"
    )
    """ parser.add_argument(
        "--out2", default="half2.mp4",
        help="Ausgabedatei für zweite Halbzeit"
    ) """
    args = parser.parse_args()

    # Schneide beide Segmente
    cut_segment(args.input, args.fh_start, args.fh_end, args.out1)
    #cut_segment(args.input, args.sh_start, args.sh_end, args.out2)

if __name__ == "__main__":
    main()