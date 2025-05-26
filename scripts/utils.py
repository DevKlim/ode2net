# File: scripts/utils.py
import os
import pretty_midi
import numpy as np # Added for clarity, though not strictly needed for current version

def tokens_to_midi(tokens: list[int], output_path: str, fs: int = 100, velocity: int = 100):
    """
    Convert a list of integer tokens into a PrettyMIDI file and write it out.
    Tokens 0-127 represent MIDI pitches.
    Token 128 (or any value >= 128) represents a rest/silence.
    Each token corresponds to a time frame of duration 1/fs seconds.

    Args:
        tokens (list[int]): List of music tokens.
        output_path (str): Path to save the output .mid file.
        fs (int): Sampling frequency (frames per second).
        velocity (int): MIDI note velocity (1-127).
    """
    if not tokens:
        print("Warning: Empty token list provided. No MIDI file will be generated.")
        return

    pm = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    
    current_note_pitch = None
    note_start_time = 0.0

    for i, token_value in enumerate(tokens):
        current_time = i / fs # Start time of the current frame

        if token_value < 128: # It's a note
            pitch = token_value
            if current_note_pitch is None: # New note starts
                current_note_pitch = pitch
                note_start_time = current_time
            elif current_note_pitch != pitch: # Current note changes pitch
                # End the previous note
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=current_note_pitch,
                    start=note_start_time,
                    end=current_time # Ends just before the new note starts
                )
                piano.notes.append(note)
                # Start the new note
                current_note_pitch = pitch
                note_start_time = current_time
            # If current_note_pitch == pitch, the note continues, do nothing yet
        
        else: # It's a rest (token_value >= 128)
            if current_note_pitch is not None: # A note was playing, now it's a rest
                # End the previous note
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=current_note_pitch,
                    start=note_start_time,
                    end=current_time # Ends just before the rest starts
                )
                piano.notes.append(note)
                current_note_pitch = None # Reset, as it's a rest
                # note_start_time is not strictly needed for rest, can be left or updated

    # After the loop, if a note was still active, end it.
    if current_note_pitch is not None:
        total_duration = len(tokens) / fs
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=current_note_pitch,
            start=note_start_time,
            end=total_duration
        )
        piano.notes.append(note)

    pm.instruments.append(piano)
    
    try:
        pm.write(output_path)
    except Exception as e:
        print(f"Error writing MIDI file to {output_path}: {e}")
        # Potentially re-raise or handle more gracefully
        raise

if __name__ == '__main__':
    # Example Usage:
    print("Running example for tokens_to_midi...")
    # Simple C Major scale, then a rest, then G note
    example_tokens = \
        [60, 62, 64, 65, 67, 69, 71, 72] * 2 + \
        [128] * 8 + \
        [67] * 8 + \
        [128] * 4 + \
        [60] * 16 
    
    output_dir = "outputs/generated_midi_examples"
    os.makedirs(output_dir, exist_ok=True)
    example_output_path = os.path.join(output_dir, "example_c_major_scale.mid")
    
    try:
        tokens_to_midi(example_tokens, example_output_path, fs=10, velocity=100) # fs=10 for shorter notes
        print(f"Example MIDI saved to {example_output_path}")
    except Exception as e:
        print(f"Error in example: {e}")

    # Example with only rests
    example_rests_tokens = [128] * 20
    example_rests_output_path = os.path.join(output_dir, "example_rests_only.mid")
    try:
        tokens_to_midi(example_rests_tokens, example_rests_output_path, fs=10)
        print(f"Example MIDI with only rests saved to {example_rests_output_path}")
    except Exception as e:
        print(f"Error in rests example: {e}")

    # Example with empty tokens
    example_empty_tokens = []
    example_empty_output_path = os.path.join(output_dir, "example_empty.mid")
    try:
        tokens_to_midi(example_empty_tokens, example_empty_output_path, fs=10)
        # This should print a warning and not create a file if handled correctly.
        if not os.path.exists(example_empty_output_path):
            print("Example with empty tokens correctly handled (no file created).")
        else:
            print(f"Warning: Empty token example created a file: {example_empty_output_path}")

    except Exception as e:
        print(f"Error in empty example: {e}")