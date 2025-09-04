import sys
from pathlib import Path

# Basis-Ordner, in dem sich deine SNMOT-*/img1-Ordner befinden
BASE_DIR = Path('../data/tracking/test')

def main():
    if not BASE_DIR.exists():
        print(f"Basis-Pfad {BASE_DIR} existiert nicht!", file=sys.stderr)
        sys.exit(1)

    total_deleted = 0
    # Alle Sequenz-Ordner finden
    for seq in sorted(BASE_DIR.glob('SNMOT-*')):
        img1 = seq / 'img1'
        if not img1.is_dir():
            print(f"  [!] {seq.name}/img1 nicht gefunden, übersprungen")
            continue

        txt_files = list(img1.glob('*.txt'))
        if not txt_files:
            print(f"  {seq.name}/img1: keine .txt-Dateien zum Löschen")
            continue

        # Löschen
        for txt in txt_files:
            try:
                txt.unlink()
                print(f"  gelöschte Datei: {txt.relative_to(BASE_DIR)}")
                total_deleted += 1
            except Exception as e:
                print(f"    Fehler beim Löschen von {txt}: {e}", file=sys.stderr)

    print(f"\nFertig! Insgesamt {total_deleted} .txt-Dateien gelöscht.")

if __name__ == '__main__':
    main()