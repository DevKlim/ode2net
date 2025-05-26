# File: scripts/generate_music.py
import os
import argparse
import torch
from dotenv import load_dotenv

# Assuming models are in a 'models' directory relative to project root
# and scripts are run from project root.
from models.text_encoder import TextEncoder
from models.music_generator import MusicGenerator
from scripts.utils import tokens_to_midi # Assuming utils.py is in the same 'scripts' directory

def main(args):
    # Load environment (e.g., for API keys if needed, though not directly used here for generation)
    load_dotenv()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Text encoder
    print("Loading Text Encoder...")
    try:
        encoder = TextEncoder(model_name=args.text_encoder_model, device=device)
    except Exception as e:
        print(f"Error loading TextEncoder model '{args.text_encoder_model}': {e}")
        return

    # Music generator model
    print("Loading Music Generator model...")
    model = MusicGenerator(
        num_tokens=args.num_tokens,  # 0-127 for pitches, 128 for rest = 129 tokens
        embedding_size=args.embedding_size, # Should match BERT's default hidden size if using its pooler_output
        num_heads=args.num_heads,
        num_layers=args.num_layers
    ).to(device)

    # Load trained weights
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return
    
    print(f"Loading model weights from {args.model_path}")
    try:
        state = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return
    model.eval()

    # Encode input text
    print(f"Encoding input text: \"{args.text[:100]}...\"") # Print a snippet
    try:
        # TextEncoder.encode returns a 1D tensor (embedding_size,)
        text_embedding = encoder.encode(args.text, max_length=args.text_max_length).to(device)
    except Exception as e:
        print(f"Error encoding text: {e}")
        return

    # Generate token sequence
    print("Generating music tokens...")
    try:
        tokens = model.generate(
            emotion_embedding=text_embedding, # Pass the 1D tensor
            start_token=args.start_token,
            max_length=args.max_length,
            temperature=args.temperature
        )
    except Exception as e:
        print(f"Error during token generation: {e}")
        return

    # Convert tokens to MIDI and write file
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_midi)
    if output_dir: # If output_midi includes a path, not just filename
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Converting {len(tokens)} tokens to MIDI...")
    try:
        tokens_to_midi(tokens, args.output_midi, fs=args.fs)
        print(f"Generated MIDI saved to {args.output_midi}")
    except Exception as e:
        print(f"Error converting tokens to MIDI or saving file: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate MIDI from text narrative using a trained model.")
    
    # Model and Generation parameters
    parser.add_argument('--model_path',   type=str, required=True,
                        help="Path to trained MusicGenerator checkpoint (.pt file)")
    parser.add_argument('--text',         type=str, required=True,
                        help="Input narrative text (enclose in quotes if it contains spaces)")
    parser.add_argument('--output_midi',  type=str, required=True,
                        help="Path to save the output .mid file")
    
    # Text Encoder parameters
    parser.add_argument('--text_encoder_model', type=str, default='bert-base-uncased',
                        help="Name of the BERT model used for text encoding (must match training).")
    parser.add_argument('--text_max_length', type=int, default=512,
                        help="Max sequence length for the text encoder.")

    # Music Generator architecture parameters (must match the trained model)
    parser.add_argument('--num_tokens', type=int, default=129,
                        help="Vocabulary size for music tokens (e.g., 128 pitches + 1 rest).")
    parser.add_argument('--embedding_size', type=int, default=768,
                        help="Embedding size in the MusicGenerator (often matches text encoder's output dim).")
    parser.add_argument('--num_heads', type=int, default=8,
                        help="Number of attention heads in the MusicGenerator's Transformer.")
    parser.add_argument('--num_layers', type=int, default=6,
                        help="Number of layers in the MusicGenerator's Transformer encoder/decoder.")

    # Generation control parameters
    parser.add_argument('--start_token',  type=int, default=128,
                        help="Start token for generation (default: 128 for rest/silence).")
    parser.add_argument('--max_length',   type=int, default=3000,
                        help="Maximum number of time-step tokens to generate (e.g., 3000 tokens at fs=100 is ~30s).")
    parser.add_argument('--temperature',  type=float, default=1.2,
                        help="Sampling temperature for token generation (higher means more randomness).")
    parser.add_argument('--fs',           type=int, default=100,
                        help="Piano-roll frame rate (frames/sec) used for MIDI conversion (must match training).")
    
    args = parser.parse_args()
    main(args)