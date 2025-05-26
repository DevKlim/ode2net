# File: scripts/preprocess_text.py
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from transformers.pipelines import pipeline
import numpy as np
from tqdm import tqdm
import re # Still useful for sanitizing parts of filenames if needed, though less critical now

# The sanitize_filename function can be kept if chapter filenames might have problematic characters,
# but if your filenames are already clean (e.g., "chapter_01.txt"), it's less crucial.
# For robustness, we can keep a simplified version or assume clean input filenames.
def sanitize_filename_component(name: str, max_len: int = 50) -> str:
    """Sanitizes a string to be a valid filename component if needed."""
    name = re.sub(r'[^\w.-]', '_', name) # Allow letters, numbers, underscore, dot, hyphen
    return name[:max_len]

def process_manual_chapters(
    input_dir: str,
    output_dir: str,
    model_name: str = 'bert-base-uncased',
    emotion_model_name: str = 'nateraw/bert-base-uncased-emotion'
):
    # The main output_dir (e.g., data/processed_text) should exist.
    # Subdirectories for books will be created as needed.
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # BERT embedding model
    print(f"Loading BERT tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name).to(device)
    bert_model.eval()

    # Emotion classifier
    print(f"Loading emotion classification pipeline: {emotion_model_name}")
    try:
        emo_pipe = pipeline(
            'text-classification',
            model=emotion_model_name,
            return_all_scores=True,
            device=0 if device == 'cuda' else -1
        )
    except Exception as e:
        print(f"Error initializing emotion pipeline with model '{emotion_model_name}': {e}. Exiting.")
        return

    # Iterate through book folders in the input_dir
    for book_folder_name in tqdm(os.listdir(input_dir), desc="Processing books"):
        book_folder_path = os.path.join(input_dir, book_folder_name)
        
        if not os.path.isdir(book_folder_path):
            # print(f"Skipping '{book_folder_name}', not a directory.")
            continue

        print(f"\nProcessing book: {book_folder_name}")

        # Create corresponding output directory for the book
        output_book_folder_path = os.path.join(output_dir, book_folder_name)
        os.makedirs(output_book_folder_path, exist_ok=True)

        # Iterate through chapter files in the current book folder
        for chapter_fn in tqdm(os.listdir(book_folder_path), desc=f"  Chapters in {book_folder_name}", leave=False):
            if not chapter_fn.endswith('.txt'):
                continue
            
            chapter_path = os.path.join(book_folder_path, chapter_fn)
            chapter_basename, _ = os.path.splitext(chapter_fn) # e.g., "chapter_01"
            
            # print(f"    Processing Chapter: {chapter_fn}")

            try:
                with open(chapter_path, 'r', encoding='utf-8') as f:
                    chapter_text = f.read()
            except Exception as e:
                print(f"    Error reading chapter file {chapter_path}: {e}. Skipping.")
                continue

            if not chapter_text.strip():
                print(f"    Chapter file {chapter_fn} is empty. Skipping.")
                continue

            # 1. Compute BERT embedding for the chapter text
            inputs = tokenizer(
                chapter_text,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                try:
                    outputs = bert_model(**inputs)
                    chapter_embedding = outputs.pooler_output.squeeze(0).cpu().numpy()
                except Exception as e:
                    print(f"    Error computing BERT embedding for chapter '{chapter_fn}': {e}. Skipping chapter.")
                    continue
            
            # 2. Compute emotion scores for the chapter text
            emotion_scores_dict = {}
            try:
                emotion_results_list = emo_pipe(chapter_text)
                if emotion_results_list and isinstance(emotion_results_list, list):
                    if len(emotion_results_list) == 1 and isinstance(emotion_results_list[0], list):
                        actual_scores = emotion_results_list[0]
                    elif all(isinstance(item, dict) for item in emotion_results_list):
                         actual_scores = emotion_results_list
                    else:
                        actual_scores = None
                    
                    if actual_scores and all(isinstance(d, dict) and 'label' in d and 'score' in d for d in actual_scores):
                        emotion_scores_dict = {item['label']: item['score'] for item in actual_scores}
                    else:
                        # print(f"    Warning: Could not parse emotion scores for chapter '{chapter_fn}'. Structure was: {actual_scores}")
                        pass # Keep emotion_scores_dict empty
                else:
                    # print(f"    Warning: Emotion pipeline returned unexpected result for chapter '{chapter_fn}': {emotion_results_list}")
                    pass # Keep emotion_scores_dict empty

            except Exception as e:
                print(f"    Error during emotion pipeline for chapter '{chapter_fn}': {e}. Emotion scores will be empty.")
            
            if not emotion_scores_dict:
                # print(f"    No valid emotion scores extracted for chapter '{chapter_fn}'. NPZ will not contain detailed emotion data.")
                emotion_labels_np = np.array([], dtype=str)
                emotion_values_np = np.array([], dtype=float)
            else:
                emotion_labels_np = np.array(list(emotion_scores_dict.keys()))
                emotion_values_np = np.array(list(emotion_scores_dict.values()))

            # 3. Save chapter embedding and emotion scores to NPZ
            output_npz_fn = f"{chapter_basename}.npz" # e.g., "chapter_01.npz"
            out_path = os.path.join(output_book_folder_path, output_npz_fn)
            
            try:
                np.savez_compressed(
                    out_path, 
                    embedding=chapter_embedding, 
                    emotion_labels=emotion_labels_np, 
                    emotion_scores=emotion_values_np,
                    book_name=np.array(book_folder_name), # Store book name
                    chapter_name=np.array(chapter_basename) # Store chapter base name
                )
                # print(f"    Saved processed chapter data to {out_path}")
            except Exception as e:
                print(f"    Error saving NPZ file {out_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess manually separated chapter text files into embeddings and emotion scores.")
    parser.add_argument('--input_dir',  type=str, default='data/raw_text',
                        help="Root directory containing book folders, which in turn contain chapter .txt files.")
    parser.add_argument('--output_dir', type=str, default='data/processed_text',
                        help="Root directory to save processed chapter .npz files, mirroring the book folder structure.")
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help="Name of the BERT model for embeddings from Hugging Face Hub.")
    parser.add_argument('--emotion_model_name', type=str, default='nateraw/bert-base-uncased-emotion',
                        help="Name of the emotion classification model from Hugging Face Hub.")
    args = parser.parse_args()
    
    process_manual_chapters(
        args.input_dir, 
        args.output_dir, 
        args.model_name, 
        args.emotion_model_name
    )