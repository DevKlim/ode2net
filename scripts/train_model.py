# File: scripts/train_model.py
import os
import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

# ... (set_seed and MusicGenerator class remain the same) ...
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class MIDIDataset(Dataset):
    def __init__(self, text_dir: str, midi_dir: str, max_seq_len: int = 1000, min_midi_frames: int = 50):
        self.max_seq_len = max_seq_len
        self.min_midi_frames = min_midi_frames
        self.text_embeddings_paths = [] # Store paths to text embedding NPZ files
        self.valid_midi_file_paths = []
        
        print(f"Scanning for text embedding .npz files in {text_dir} (recursive)...")
        # Recursively find all .npz files in text_dir and its subdirectories
        for root, _, files in os.walk(text_dir):
            for file in files:
                if file.endswith('.npz'):
                    self.text_embeddings_paths.append(os.path.join(root, file))
        
        if not self.text_embeddings_paths:
            raise ValueError(f"No text embedding .npz files found in {text_dir} or its subdirectories. "
                             "Ensure 'data/processed_text/' contains .npz files, possibly in subfolders per book. "
                             "You might need to re-run the (modified) preprocess_text.py script.")
        print(f"Found {len(self.text_embeddings_paths)} text embedding files.")

        # MIDI file loading remains the same (assumes MIDI files are directly in midi_dir, not nested)
        print(f"Scanning and pre-validating MIDI files from {midi_dir}...")
        midi_files_in_dir = [f for f in os.listdir(midi_dir) if f.endswith('.npz')]
        for fn_midi_npz in tqdm(midi_files_in_dir, desc="Pre-validating MIDI files"):
            midi_path = os.path.join(midi_dir, fn_midi_npz)
            try:
                with np.load(midi_path) as midi_data_peek:
                    if 'piano_roll' not in midi_data_peek:
                        continue
                    if midi_data_peek['piano_roll'].shape[1] < self.min_midi_frames:
                        continue
                self.valid_midi_file_paths.append(midi_path)
            except Exception as e:
                continue

        if not self.valid_midi_file_paths:
            raise ValueError(f"No valid MIDI .npz files found or loaded from {midi_dir} meeting criteria. "
                             "Ensure 'data/processed_midi/' is populated and files are not too short.")
        print(f"Found {len(self.valid_midi_file_paths)} valid processed MIDI files.")

    def __len__(self):
        return len(self.valid_midi_file_paths) # Or len(self.text_embeddings_paths) if you want to pair each text with a random MIDI

    def __getitem__(self, idx):
        # Load a random text embedding
        # This ensures that even if num_text_files != num_midi_files, we can still form pairs.
        # If you want a fixed pairing, you'd need to adjust the logic and __len__.
        text_embedding_path = random.choice(self.text_embeddings_paths)
        try:
            text_data = np.load(text_embedding_path, allow_pickle=True)
            if 'embedding' not in text_data:
                raise KeyError(f"'embedding' key not found in {text_embedding_path}")
            selected_text_embedding = text_data['embedding'].astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Error loading text embedding from {text_embedding_path}: {e}")


        # Load MIDI data (same as before)
        midi_path = self.valid_midi_file_paths[idx % len(self.valid_midi_file_paths)] # Use modulo if lengths differ
        try:
            midi_data = np.load(midi_path)
            piano_roll = midi_data['piano_roll']
        except Exception as e:
            raise RuntimeError(f"Error loading MIDI data from {midi_path} during __getitem__: {e}")

        token_sequence = []
        for frame_idx in range(piano_roll.shape[1]):
            frame_pitches = piano_roll[:, frame_idx]
            active_pitches = np.where(frame_pitches > 0)[0]
            token = int(active_pitches[0]) if len(active_pitches) > 0 else 128
            token_sequence.append(token)
        
        if len(token_sequence) > self.max_seq_len:
            token_sequence = token_sequence[:self.max_seq_len]
        else:
            token_sequence += [128] * (self.max_seq_len - len(token_sequence))
        
        full_sequence_np = np.array(token_sequence, dtype=np.int64)
        input_tokens = full_sequence_np[:-1]
        target_tokens = full_sequence_np[1:]
        
        return selected_text_embedding, input_tokens, target_tokens

# ... (train function and main block remain largely the same) ...
# Ensure MusicGenerator class is defined or imported correctly
class MusicGenerator(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        embedding_size: int = 768,
        num_heads: int = 8,
        num_layers: int = 6
    ):
        super(MusicGenerator, self).__init__()
        self.embedding = nn.Embedding(num_tokens, embedding_size)
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            num_encoder_layers=num_layers, # Transformer expects encoder and decoder layers
            num_decoder_layers=num_layers
        )
        self.fc_out = nn.Linear(embedding_size, num_tokens)

    def forward(self, emotion_embedding: torch.Tensor, music_tokens: torch.Tensor) -> torch.Tensor:
        src = emotion_embedding.unsqueeze(0)
        tgt = self.embedding(music_tokens)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        # The nn.Transformer module by default has an encoder and a decoder.
        # For conditional generation where `src` is the condition and `tgt` is the sequence to generate,
        # `src` goes to the encoder and `tgt` to the decoder.
        # If your MusicGenerator is purely a decoder (like GPT), you'd use nn.TransformerDecoder.
        # Assuming standard Transformer:
        output = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask)
        logits = self.fc_out(output)
        return logits

def train(args):
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Initializing dataset and dataloader...")
    try:
        dataset = MIDIDataset(
            args.text_dir, 
            args.midi_dir, 
            max_seq_len=args.max_seq_len,
            min_midi_frames=args.min_midi_frames
        )
    except ValueError as e:
        print(f"Error initializing dataset: {e}")
        return 

    if len(dataset) == 0: 
        print("Dataset is empty. Aborting training.")
        return
        
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True if device=='cuda' else False,
        drop_last=True 
    )

    print("Initializing MusicGenerator model...")
    model = MusicGenerator(
        num_tokens=args.num_tokens,
        embedding_size=args.embedding_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=128 if args.ignore_rest_in_loss else -100) 

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for emb_batch, input_seq_batch, target_seq_batch in progress_bar:
            emb_batch = emb_batch.to(device)
            input_tokens_model = input_seq_batch.to(device).transpose(0, 1)
            target_tokens_model = target_seq_batch.to(device).transpose(0, 1)

            optimizer.zero_grad()
            logits = model(emb_batch, input_tokens_model)
            loss = loss_fn(logits.reshape(-1, args.num_tokens), target_tokens_model.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = running_loss / len(dataloader)
        print(f"[Epoch {epoch}/{args.epochs}] Average Training Loss: {avg_loss:.4f}")

    output_model_dir = os.path.dirname(args.save_path)
    if output_model_dir and not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir, exist_ok=True)
    
    try:
        torch.save(model.state_dict(), args.save_path)
        print(f"Model checkpoint saved to {args.save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an emotion-conditioned MIDI generator model.")
    
    parser.add_argument('--text_dir',    type=str, default='data/processed_text',
                        help="Root directory containing processed text .npz files (embeddings for text chapters), possibly in subfolders per book.")
    parser.add_argument('--midi_dir',    type=str, default='data/processed_midi',
                        help="Directory containing processed MIDI .npz files (piano rolls). Assumed to be flat, not nested.")
    parser.add_argument('--max_seq_len', type=int, default=1000, 
                        help="Maximum sequence length for MIDI tokens (truncate or pad).")
    parser.add_argument('--min_midi_frames', type=int, default=50, 
                        help="Minimum number of frames for a MIDI file to be included in training.")

    parser.add_argument('--batch_size',  type=int, default=16, help="Batch size for training.")
    parser.add_argument('--lr',          type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay for Adam optimizer.")
    parser.add_argument('--epochs',      type=int, default=20, help="Number of training epochs.")
    parser.add_argument('--num_workers', type=int, default=0, 
                        help="Number of worker processes for DataLoader.")
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help="Max norm for gradient clipping.")
    parser.add_argument('--ignore_rest_in_loss', action='store_true',
                        help="If set, rest tokens (128) will be ignored in the loss calculation.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")

    parser.add_argument('--num_tokens', type=int, default=129, 
                        help="Vocabulary size for music tokens.")
    parser.add_argument('--embedding_size', type=int, default=768, 
                        help="Embedding size in the MusicGenerator (should match text encoder's output dim).")
    parser.add_argument('--num_heads', type=int, default=8,
                        help="Number of attention heads in the MusicGenerator's Transformer.")
    parser.add_argument('--num_layers', type=int, default=6,
                        help="Number of layers in the MusicGenerator's Transformer encoder/decoder.")

    parser.add_argument('--save_path',   type=str, default='models/music_generator_checkpoint.pt',
                        help="Path to save the trained model checkpoint.")
    
    args = parser.parse_args()
    train(args)