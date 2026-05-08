#!/usr/bin/env python3
"""
audiosentinel — CLI for Human vs AI audio detection

Usage:
    audiosentinel recording.wav
    audiosentinel audio/*.wav
"""
import sys
import os


def main():
    if len(sys.argv) < 2:
        print("Usage: audiosentinel <file.wav> [file2.wav ...]")
        sys.exit(1)

    from audiosentinel import predict_audio

    paths = sys.argv[1:]
    failed = 0

    for path in paths:
        if not os.path.exists(path):
            print(f"✗ Not found: {path}")
            failed += 1
            continue
        try:
            result = predict_audio(path, verbose=False)
            icon   = "👤" if result['pred'] == 1 else "🤖"
            conf   = max(result['prob_ai'], result['prob_human']) * 100
            print(f"{icon}  {result['label']:5s}  {conf:5.1f}%  {os.path.basename(path)}")
        except Exception as e:
            print(f"✗ Failed: {path} — {e}")
            failed += 1

    if len(paths) > 1:
        print(f"\nProcessed {len(paths) - failed}/{len(paths)} files")


if __name__ == '__main__':
    main()
              
