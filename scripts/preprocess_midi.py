# File: scripts/preprocess_midi.py
import os
import argparse
import numpy as np
import pretty_midi
from tqdm import tqdm
import random # For random sampling

def process_midi(
    input_dir: str,
    output_dir: str,
    fs: int = 100,
    num_aria_files: int = 5000,
    num_lakh_files: int = 5000,
    random_seed: int = 42 # For reproducibility of sampling
):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(random_seed) # Set seed for random sampling

    aria_midi_files = []
    lakh_midi_files = []
    other_midi_files = [] # For any files not in 'aria_midi' or 'lakh_clean' subdirs

    print(f"Scanning for MIDI files in {input_dir}...")
    # Collect all MIDI files and categorize them
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.lower().endswith(('.mid', '.midi')):
                path = os.path.join(root, fn)
                # Categorize based on parent directory
                # This assumes 'aria_midi' and 'lakh_clean' are direct subdirectories of input_dir
                # or identifiable parts of the path.
                if 'aria_midi' in root.lower(): # Check if 'aria_midi' is part of the path
                    aria_midi_files.append(path)
                elif 'lakh_clean' in root.lower(): # Check if 'lakh_clean' is part of the path
                    lakh_midi_files.append(path)
                else:
                    other_midi_files.append(path)
    
    print(f"Found {len(aria_midi_files)} ARIA MIDI files.")
    print(f"Found {len(lakh_midi_files)} Lakh MIDI files.")
    print(f"Found {len(other_midi_files)} other MIDI files.")

    # Randomly sample from ARIA files
    if len(aria_midi_files) > num_aria_files:
        selected_aria_files = random.sample(aria_midi_files, num_aria_files)
        print(f"Randomly selected {len(selected_aria_files)} ARIA MIDI files.")
    else:
        selected_aria_files = aria_midi_files # Take all if fewer than requested
        print(f"Selected all {len(selected_aria_files)} available ARIA MIDI files (less than requested {num_aria_files}).")

    # Randomly sample from Lakh files
    if len(lakh_midi_files) > num_lakh_files:
        selected_lakh_files = random.sample(lakh_midi_files, num_lakh_files)
        print(f"Randomly selected {len(selected_lakh_files)} Lakh MIDI files.")
    else:
        selected_lakh_files = lakh_midi_files # Take all if fewer than requested
        print(f"Selected all {len(selected_lakh_files)} available Lakh MIDI files (less than requested {num_lakh_files}).")

    # Combine selected files (and include 'other' files if desired, or ignore them)
    # For now, let's only process the selected ARIA and Lakh files.
    # If you want to include 'other_midi_files', add them to file_list_to_process.
    file_list_to_process = selected_aria_files + selected_lakh_files
    
    if not file_list_to_process:
        print(f"No MIDI files selected for processing. Exiting.")
        return

    print(f"\nTotal {len(file_list_to_process)} MIDI files selected for processing.")
    processed_count = 0
    skipped_count = 0

    for path in tqdm(file_list_to_process, desc="Processing selected MIDI files"):
        fn = os.path.basename(path)
        try:
            midi = pretty_midi.PrettyMIDI(path)
        except Exception as e:
            # print(f"Skipping {fn} due to error during PrettyMIDI loading: {e}")
            skipped_count += 1
            continue

        if not midi.instruments:
            # print(f"Skipping {fn}: No instruments found in MIDI file.")
            skipped_count += 1
            continue

        total_notes = sum(len(instr.notes) for instr in midi.instruments)
        if total_notes == 0:
            # print(f"Skipping {fn}: No notes found in any instrument.")
            skipped_count += 1
            continue
            
        total_time = midi.get_end_time()
        if total_time <= 0:
            # print(f"Skipping {fn}: Invalid or zero total time ({total_time:.2f}s).")
            skipped_count += 1
            continue
        
        note_density = total_notes / total_time

        try:
            piano_roll = midi.get_piano_roll(fs=fs) 
            piano_roll_binary = (piano_roll > 0).astype(np.uint8)
        except Exception as e:
            # print(f"Skipping {fn}: Error generating piano roll: {e}")
            skipped_count += 1
            continue
        
        if piano_roll_binary.shape[1] == 0: 
            # print(f"Skipping {fn}: Piano roll has zero frames (length).")
            skipped_count += 1
            continue

        base_fn, _ = os.path.splitext(fn)
        # To avoid name collisions if ARIA and Lakh have files with the same name,
        # we can add a prefix to the output filename.
        path_parts = path.split(os.sep)
        prefix = ""
        if 'aria_midi' in path_parts: # Check if 'aria_midi' is in the full path
            prefix = "aria_"
        elif 'lakh_clean' in path_parts: # Check if 'lakh_clean' is in the full path
            prefix = "lakh_"
        
        out_fn = f"{prefix}{base_fn}.npz"
        
        try:
            np.savez_compressed(
                os.path.join(output_dir, out_fn),
                piano_roll=piano_roll_binary,
                note_density=np.float32(note_density)
            )
            processed_count += 1
        except Exception as e:
            print(f"Error saving NPZ for {fn}: {e}")
            skipped_count += 1
            
    print(f"\nMIDI preprocessing finished.")
    print(f"Successfully processed: {processed_count} files.")
    print(f"Skipped: {skipped_count} files due to errors or invalid format.")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Preprocess a random subset of MIDI files into piano rolls.")
    p.add_argument('--input_dir',  type=str, default='data/raw_midi',
                        help="Root directory containing raw MIDI files. Expects subdirectories like 'aria_midi' and 'lakh_clean'.")
    p.add_argument('--output_dir', type=str, default='data/processed_midi',
                        help="Directory to save processed .npz files.")
    p.add_argument('--fs',         type=int, default=100,
                        help="Sampling frequency for the piano roll (frames per second).")
    p.add_argument('--num_aria_files', type=int, default=5000,
                        help="Number of MIDI files to randomly select from the 'aria_midi' subdirectory.")
    p.add_argument('--num_lakh_files', type=int, default=5000,
                        help="Number of MIDI files to randomly select from the 'lakh_clean' subdirectory.")
    p.add_argument('--random_seed', type=int, default=42,
                        help="Random seed for reproducible sampling.")
    args = p.parse_args() # Corrected: use p here
    
    print(f"Starting MIDI preprocessing from '{args.input_dir}' to '{args.output_dir}' with fs={args.fs}.")
    print(f"Will attempt to select {args.num_aria_files} ARIA files and {args.num_lakh_files} Lakh files.")
    process_midi(
        args.input_dir, 
        args.output_dir, 
        fs=args.fs, 
        num_aria_files=args.num_aria_files,
        num_lakh_files=args.num_lakh_files,
        random_seed=args.random_seed
    )