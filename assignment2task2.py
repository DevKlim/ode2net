# %%
# %%
# Cell 1: Pipeline class + MIDI‐emotion helpers (standalone)
# (MULTI-TRACK V4 - Expanded Emotions, Enhanced Legato & Rhythmic Variety)

from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy, textstat, random, mido, re, shutil, glob, os, stat, pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

class FormAwareTextToMusic:
    def __init__(self, base_tempo=80, tpb=480, cycles=4):
        self.emoc_model_name = "j-hartmann/emotion-english-distilroberta-base"
        self.emoc = pipeline("text-classification", model=self.emoc_model_name, return_all_scores=True)
        self.known_emotions_from_model = list(self.emoc.model.config.label2id.keys())
        
        self.zs_topic = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.topics = ["love","betrayal","death","fate","family","violence", "mystery", "peace", "tension", "hope", "despair", "reflection"]
        self.zs_form = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.forms = ["sonata","etude","prelude","waltz","nocturne","fugue", "ballad", "fantasy", "elegy", "rhapsody"]
        self.tfidf = TfidfVectorizer(max_features=5, stop_words="english", min_df=1)
        self.nlp = spacy.load("en_core_web_sm")
        self.base_tempo = base_tempo
        self.tpb = tpb 
        self.cycles = cycles

        # Added more moods, legato_intensity, base_note_duration_mult
        self.MUSICAL_MOOD_PARAMS = { 
          "joy":        {"tempo":+20, "vol":(60,90),"mel_adj":0,  "mel_range": (2, 6), "r_complex":0.7, "prog": ["I", "V", "vi", "IV"], "dyn_shape": "crescendo", "legato_i":0.85, "base_dur_mult":1.0, "is_major":True},
          "sadness":    {"tempo":-30, "vol":(30,55), "mel_adj":-15,"mel_range": (1, 2), "r_complex":0.15,"prog": ["i", "iv", "VI", "V"], "dyn_shape": "diminuendo","legato_i":0.99, "base_dur_mult":2.0, "is_major":False}, # Longer notes
          "anger":      {"tempo":+10, "vol":(80,100),"mel_adj":5,  "mel_range": (2, 7), "r_complex":0.65,"prog": ["i", "VI", "bII", "V"], "dyn_shape": "accented",  "legato_i":0.7,  "base_dur_mult":0.8, "is_major":False}, # Neapolitan for anger
          "fear":       {"tempo": -5, "vol":(40,70), "mel_adj":-10,"mel_range": (1, 4), "r_complex":0.25,"prog": ["i", "iidim", "VI", "V"], "dyn_shape": "subdued_var","legato_i":0.9,  "base_dur_mult":1.5, "is_major":False},
          "surprise":   {"tempo":+15, "vol":(70,100),"mel_adj":5,  "mel_range": (3, 8), "r_complex":0.8, "prog": ["I", "bVI", "IV", "V"], "dyn_shape": "sudden_loud","legato_i":0.75, "base_dur_mult":0.9, "is_major":True},
          "love":       {"tempo":-10,  "vol":(40,70), "mel_adj":-5, "mel_range": (2, 4), "r_complex":0.4, "prog": ["I", "vi", "ii", "V"], "dyn_shape": "gentle_swell","legato_i":0.95, "base_dur_mult":1.2, "is_major":True},
          "neutral":    {"tempo":  0, "vol":(50,75), "mel_adj":-5, "mel_range": (2, 5), "r_complex":0.5, "prog": ["I", "IV", "V", "ii"], "dyn_shape": "flat",      "legato_i":0.85, "base_dur_mult":1.0, "is_major":True},
          "serenity":   {"tempo":-35, "vol":(20,40), "mel_adj":-20,"mel_range": (1, 2), "r_complex":0.1, "prog": ["Imaj7", "IVmaj7", "ii7", "V7sus"], "dyn_shape": "very_flat_soft","legato_i":0.99, "base_dur_mult":2.5, "is_major":True}, # Very long notes
          "suspense":   {"tempo":-15, "vol":(35,60), "mel_adj":-15,"mel_range": (1, 3), "r_complex":0.15,"prog": ["i", "bII", "Vaug", "i(add b6)"], "dyn_shape": "trembling", "legato_i":0.8, "base_dur_mult":1.8, "is_major":False}, # Dissonant
          "anticipation":{"tempo":+10, "vol":(50,80), "mel_adj":-5, "mel_range": (2,5), "r_complex":0.4, "prog": ["IV", "V", "IV/vi", "V7"], "dyn_shape":"crescendo_slight", "legato_i":0.8, "base_dur_mult":1.0, "is_major":True}, # Unresolved tension
          "trust":      {"tempo": -10, "vol":(55,80), "mel_adj":-5, "mel_range": (1,3), "r_complex":0.3, "prog": ["I", "IV", "V", "I"], "dyn_shape":"flat_warm", "legato_i":0.9, "base_dur_mult":1.5, "is_major":True}, # Stable, warm
          "awe":        {"tempo":-10, "vol":(60,90), "mel_adj":0,  "mel_range": (3,7), "r_complex":0.25,"prog": ["I", "VI", "IV", "bVII"], "dyn_shape":"grand_swell", "legato_i":0.95, "base_dur_mult":2.0, "is_major":True}, # Wider leaps, slower
          "remorse":    {"tempo":-20, "vol":(30,60), "mel_adj":-10,"mel_range": (1,2), "r_complex":0.2, "prog": ["i", "iv", "iidim", "V"], "dyn_shape":"diminuendo_slow", "legato_i":0.98, "base_dur_mult":1.8, "is_major":False},
          "optimism":   {"tempo":+10, "vol":(65,95), "mel_adj":0,  "mel_range": (2,5), "r_complex":0.6, "prog": ["I", "IV", "V/V", "V"], "dyn_shape":"crescendo_bright", "legato_i":0.85, "base_dur_mult":1.0, "is_major":True},
          "pessimism":  {"tempo":-25, "vol":(40,65), "mel_adj":-8, "mel_range": (1,3), "r_complex":0.2, "prog": ["i", "bVI", "iv", "V"], "dyn_shape":"diminuendo_heavy", "legato_i":0.95, "base_dur_mult":1.7, "is_major":False},
        }
        # Shortened key names for brevity in MUSICAL_MOOD_PARAMS
        # vol = base volume range for accompaniment, mel_adj = melody volume adjustment relative to this
        # mel_range = melodic range hint, r_complex = rhythmic complexity
        # prog = chord progression, dyn_shape = dynamic shape hint
        # legato_i = legato intensity for melody, base_dur_mult = base note duration multiplier

        self.TOPIC_PARAMS = {
          "love":1.2,"betrayal":0.8,"death":0.6, "mystery": 0.9, "peace": 1.1, "tension":0.7,
          "hope": 1.15, "despair":0.7, "reflection":0.9,
          "fate":1.0,"family":1.1,"violence":0.7,"neutral":1.0
        }
        self.MAJOR_DIATONIC_CHORDS = ["M", "m", "m", "M", "M", "m", "d"]
        self.MINOR_DIATONIC_CHORDS = ["m", "d", "M", "m", "m", "M", "M"]

    def _derive_musical_mood(self, raw_emo_label, emo_scores, topic, text_keywords):
        # emo_scores is a dict like {'joy': 0.9, 'sadness': 0.05, ...}
        # More sophisticated mapping
        if raw_emo_label == "joy":
            if topic == "love": return "love"
            if "hope" in text_keywords or topic == "hope": return "optimism"
            return "joy"
        if raw_emo_label == "sadness":
            if "sorry" in text_keywords or "regret" in text_keywords : return "remorse"
            if topic == "despair": return "pessimism"
            if topic == "peace" or "calm" in text_keywords: return "serenity"
            return "sadness"
        if raw_emo_label == "fear":
            if topic in ["mystery", "tension"]: return "suspense"
            if emo_scores.get("surprise", 0) > 0.3: return "awe" # Fear + Surprise -> Awe
            return "fear"
        if raw_emo_label == "surprise":
            if emo_scores.get("joy",0) > 0.3: return "awe" # Surprise + Joy -> Awe (more positive)
            if topic == "hope": return "optimism"
            return "surprise"
        if raw_emo_label == "anger":
            # Could check for "betrayal" topic to make it more specific if we had a "betrayal" mood
            return "anger"
        if raw_emo_label == "neutral":
            if topic == "peace" or "calm" in text_keywords: return "serenity"
            if topic == "reflection": return "trust" # Trust can be a calm, reflective mood
            return "neutral"
        if raw_emo_label == "disgust": # Model's 'disgust'
            if emo_scores.get("sadness",0) > 0.3: return "remorse"
            return "anger" # Default map for disgust

        # Fallback for any other model outputs or unmapped scenarios
        return "neutral"


    def analyze_text(self, text):
        model_output = self.emoc(text[:512])[0] # Get all scores from the list
        # Create a dictionary of scores for easier access
        emo_scores_dict = {item['label'].lower(): item['score'] for item in model_output}
        raw_emo_label = max(emo_scores_dict, key=emo_scores_dict.get) # Dominant raw emotion
        
        topic = self.zs_topic(text[:512], candidate_labels=self.topics)["labels"][0].lower()
        form_label  = self.zs_form(text[:512], candidate_labels=self.forms)["labels"][0].lower()
        
        try:
            self.tfidf.fit([text])
            kws = list(self.tfidf.get_feature_names_out())
        except: kws = []
        
        musical_mood_key = self._derive_musical_mood(raw_emo_label, emo_scores_dict, topic, kws)
        mood_params = self.MUSICAL_MOOD_PARAMS.get(musical_mood_key, self.MUSICAL_MOOD_PARAMS["neutral"])
        
        tmult = self.TOPIC_PARAMS.get(topic,1.0)
        doc = self.nlp(text[:5000]) 
        ne_ratio  = len(doc.ents)/max(len(doc),1) if doc and len(doc) > 0 else 0
        quotes    = len(re.findall(r'“[^”]+”', text))
        num_sents = len(list(doc.sents)) if doc else 0
        dlg_ratio = quotes/max(num_sents,1) if num_sents > 0 else 0
        read     = textstat.flesch_reading_ease(text)
        read_adj = int((read-60)/10)
        
        tempo = max(30, min(180, self.base_tempo + mood_params["tempo"] + read_adj))
        base_density_factor = tmult * (1 + dlg_ratio) 
        r_complex_factor = mood_params.get("r_complex", 0.5)
        # Adjust density to allow for very low values for long notes
        density = np.clip(base_density_factor * (0.5 + r_complex_factor), 0.15, 4.0) 

        vol_lo_accomp, vol_hi_accomp = mood_params["vol"]
        mel_vol_adj = mood_params.get("mel_adj", 0)
        melody_specific_vol_range = (max(10, vol_lo_accomp + mel_vol_adj), max(20, vol_hi_accomp + mel_vol_adj))
        accompaniment_base_vol_range = mood_params["vol"]

        melodic_range_hint = mood_params.get("mel_range", (2,5)) 
        chord_progression_template = mood_params.get("prog", ["I", "IV", "V", "I"])
        dynamic_shape_hint = mood_params.get("dyn_shape", "flat")
        legato_intensity = mood_params.get("legato_i", 0.85) 
        base_note_duration_multiplier = mood_params.get("base_dur_mult", 1.0)

        if "is_major" in mood_params: is_major_mode = mood_params["is_major"]
        else: is_major_mode = musical_mood_key in ("joy","surprise","love","neutral","serenity","trust","awe","optimism")

        mode_intervals = [0,2,4,5,7,9,11] if is_major_mode else [0,2,3,5,7,8,10]
        shift = (sum(hash(k) for k in kws)+int(ne_ratio*12)) % 12 
        melody_scale_notes = sorted([60 + i + shift for i in mode_intervals] + [72 + i + shift for i in mode_intervals])
        harmony_base_octave = 48 
        harmony_scale_root_notes = sorted([harmony_base_octave + i + shift for i in mode_intervals])

        return (musical_mood_key, topic, form_label, melody_scale_notes, tempo, density, 
                melody_specific_vol_range, accompaniment_base_vol_range, melodic_range_hint,
                chord_progression_template, harmony_scale_root_notes, is_major_mode, dynamic_shape_hint,
                legato_intensity, base_note_duration_multiplier) # Now 15 params

    # _get_velocity_for_shape, _select_next_note_index, _get_triad_notes, _map_roman_to_scale_degree
    # (These remain largely the same as V3, minor tweaks if needed, e.g. _get_velocity_for_shape for new dyn_shapes)
    def _get_velocity_for_shape(self, base_vol_range, progress_in_phrase, dynamic_shape_hint):
        lo, hi = base_vol_range
        span = hi - lo
        vel = random.randint(lo, hi) # Default
        
        if dynamic_shape_hint == "crescendo" or dynamic_shape_hint == "crescendo_slight" or dynamic_shape_hint == "crescendo_bright":
            vel = int(lo + span * progress_in_phrase)
        elif dynamic_shape_hint == "diminuendo" or dynamic_shape_hint == "diminuendo_slow" or dynamic_shape_hint == "diminuendo_heavy":
            vel = int(hi - span * progress_in_phrase)
        elif dynamic_shape_hint == "gentle_swell" or dynamic_shape_hint == "grand_swell": 
            vel = int(lo + span * (0.5 + 0.5 * np.sin((progress_in_phrase - 0.5) * np.pi)))
        elif dynamic_shape_hint == "accented": 
            vel = int(hi - span * progress_in_phrase * 0.7) if progress_in_phrase > 0.1 else hi
        elif dynamic_shape_hint == "sudden_loud": 
            vel = hi if progress_in_phrase > 0.8 else int(lo + span * 0.2)
        elif dynamic_shape_hint == "subdued_var": 
            vel = int(lo + span * (0.1 + 0.2 * np.sin(progress_in_phrase * 4 * np.pi)))
        elif dynamic_shape_hint == "very_flat_soft":
            vel = int(lo + span * 0.1) 
        elif dynamic_shape_hint == "trembling": 
            vel = int(lo + span * (0.1 + 0.1 * random.choice([-1,1,0,-0.5,0.5])))
        elif dynamic_shape_hint == "flat_warm":
            vel = int(lo + span * 0.4) # Consistently warm mid-volume
        # else "flat" uses the random.randint default
        return max(10, min(127, vel)) 

    def _select_next_note_index(self, current_note_idx, scale_len, melodic_range_hint, prev_step=0, tendency=0, mood_key="neutral"):
        min_jump, max_jump = melodic_range_hint
        
        possible_intervals = [-2, -1, 1, 2] 
        # For very smooth moods, mostly stepwise
        if mood_key in ["serenity", "sadness", "remorse", "pessimism"]:
            max_jump = min(max_jump, 2) # Limit jumps for these moods

        for j in range(3, max_jump + 1):
            if random.random() < 0.3 / (j+1e-6) : possible_intervals.extend([-j, j]) 

        weights = []
        for interval in possible_intervals:
            w = 1.0 / (abs(interval) + 0.1) 
            if abs(interval) > min_jump : w *= 0.4 # Stronger penalty for jumps
            if abs(interval) == 1 and mood_key in ["serenity", "sadness", "remorse"]: w *= 2.0 # Favor stepwise for calm/sad
            if interval == -prev_step and abs(prev_step) > 1: w*= 0.2 
            if tendency != 0 and np.sign(interval) == tendency: w *= 1.5 
            weights.append(max(0.01, w)) # Ensure weight is positive
        
        if not possible_intervals or not any(w > 0 for w in weights): 
             return (current_note_idx + 1) % scale_len, 1

        chosen_step = random.choices(possible_intervals, weights=weights, k=1)[0]
        next_idx = current_note_idx + chosen_step
        next_idx = np.clip(next_idx, 0, scale_len - 1)
        
        if next_idx == current_note_idx and scale_len > 1:
            # Try to move if stuck, respecting tendency if possible
            if tendency > 0: chosen_step = 1
            elif tendency < 0: chosen_step = -1
            else: chosen_step = random.choice([-1,1])
            next_idx = (current_note_idx + chosen_step + scale_len) % scale_len
            
        return int(next_idx), chosen_step

    def _get_triad_notes(self, root_note_pitch, is_major_key, chord_type_str, scale_intervals, 
                         mood_key="neutral", current_roman_numeral_in_prog="I", root_degree_idx_for_chord=0):
        # Allow mood to influence chord voicing slightly (e.g. add 7ths for some moods)
        # Basic triads for now
        third_interval = 4 if chord_type_str == "M" else 3 
        if chord_type_str == "A": # Augmented
            fifth_interval = 8
        elif chord_type_str == "d": 
            fifth_interval = 6 
        else: # Major or minor chord (perfect fifth)
            fifth_interval = 7 
        
        notes = [root_note_pitch, root_note_pitch + third_interval, root_note_pitch + fifth_interval]
        
        # Simple 7th for specific moods/chords if desired
        # For Imaj7 in Serenity/Love
        if mood_key in ["serenity", "love"] and chord_type_str == "M" and \
           ("maj7" in current_roman_numeral_in_prog.lower() or (root_degree_idx_for_chord == 0 and random.random() < 0.4)): # I or Imaj7
            notes.append(root_note_pitch + 11) # Major 7th

        # For V7 (Dominant 7th)
        elif chord_type_str == "M" and root_degree_idx_for_chord == 4 and \
             ("7" in current_roman_numeral_in_prog and "maj7" not in current_roman_numeral_in_prog.lower()): # If it's a V chord and "7" is in the roman numeral (but not maj7)
            notes.append(root_note_pitch + 10) # Minor 7th for V7
        
        # For ii7 or other minor 7ths if specified or for suspense
        elif chord_type_str == "m" and \
             ("7" in current_roman_numeral_in_prog or (mood_key == "suspense" and random.random() < 0.3)):
            notes.append(root_note_pitch + 10) # Minor 7th

        return [n for n in notes if 20 < n < 109]

    def _map_roman_to_scale_degree(self, roman_numeral_str):
        # ... (same as before, ensure it handles new suffixes from progressions)
        cleaned_roman = roman_numeral_str.lower().replace("dim","").replace("o","").replace("aug","").replace("+","").replace("maj7","").replace("7sus","").replace("7","").replace("(add b6)","")
        mapping = {"i":0, "I":0, "ii":1, "II":1, "iii":2, "III":2, "iv":3, "IV":3, 
                   "v":4, "V":4, "vi":5, "VI":5, "vii":6, "VII":6,
                   "bii":1, "bVI":5, "bVII":6} # For Neapolitan, etc.
        # Find the core Roman numeral part
        core_match = re.match(r"^(b?)([ivxIVX]+)", cleaned_roman)
        if core_match:
            numeral_part = core_match.group(2)
            degree = {"i":0,"ii":1,"iii":2,"iv":3,"v":4,"vi":5,"vii":6}.get(numeral_part,0)
            # Basic handling for b (flat) modifiers, assumes it's a common flat like bII, bVI, bVII
            # This simplified mapping may need adjustment for more complex theory
            return degree
        return 0 # Default


    def _build_multi_track_music(self, musical_mood, topic, form, melody_scale_notes, tempo, density, 
                                 melody_vol_range, accompaniment_base_vol_range, 
                                 melodic_range_hint, chord_progression_template,
                                 harmony_scale_root_notes, is_major_mode, dynamic_shape_hint,
                                 legato_intensity, base_note_duration_multiplier): # Added base_note_duration_multiplier
        # ... (initial setup from V3: mid, tracks, program changes, time_sig, tempo) ...
        mid = mido.MidiFile(ticks_per_beat=self.tpb)
        melody_track = mido.MidiTrack(); melody_track.name = "Melody (Ch 0)"; mid.tracks.append(melody_track)
        bass_track = mido.MidiTrack(); bass_track.name = "Bass (Ch 1)"; mid.tracks.append(bass_track)
        chord_track = mido.MidiTrack(); chord_track.name = "Chords (Ch 2)"; mid.tracks.append(chord_track)

        melody_track.append(mido.Message('program_change', channel=0, program=0, time=0)) # Piano
        # Consider changing instruments based on mood
        if musical_mood in ["serenity", "awe"]:
            chord_track.append(mido.Message('program_change', channel=2, program=52, time=0)) # Choir Aahs / Synth Voice
        elif musical_mood == "suspense":
             chord_track.append(mido.Message('program_change', channel=2, program=99, time=0)) # FX 3 (crystal) or 50 (Synth Strings 1)
        else:
            chord_track.append(mido.Message('program_change', channel=2, program=48, time=0)) # Strings
        bass_track.append(mido.Message('program_change', channel=1, program=33, time=0))


        is_waltz = (form == "waltz")
        beats_per_measure = 3 if is_waltz else 4
        current_tick_abs = 0 
        melody_track.append(mido.MetaMessage('time_signature', numerator=beats_per_measure, denominator=4, time=0))
        melody_track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))

        if not melody_scale_notes: melody_scale_notes = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83] 
        melody_scale_len = len(melody_scale_notes)
        current_melody_note_idx = random.randint(melody_scale_len // 4, melody_scale_len // 2) 
        prev_melody_step = 0 

        # Apply base_note_duration_multiplier to the average slot duration
        avg_rhythmic_slot_duration_ticks = (self.tpb / density) * base_note_duration_multiplier
        # Ensure it's not excessively small
        avg_rhythmic_slot_duration_ticks = max(self.tpb / 8, avg_rhythmic_slot_duration_ticks) # at least a 32nd note for base if tpb=480

        rhythmic_patterns = {
            "simple": [1.0], "long_focus": [2.0, 1.0, 1.0] if base_note_duration_multiplier >=1.5 else [1.0], # if base is already long, keep it simple
            "varied1": [1.5, 0.5], "varied2": [0.75, 0.25, 1.0],"dotted": [0.75,0.25],
            "sparse": [2.0, 2.0] if base_note_duration_multiplier >=1.5 else [1.0,1.0,2.0], # For very slow moods
            "even_flow": [0.5, 0.5] if density * base_note_duration_multiplier >=1.0 else [1.0] 
        }
        if musical_mood in ["serenity", "sadness", "pessimism", "remorse"]: chosen_rhythm_key_default = "sparse"
        elif musical_mood == "suspense": chosen_rhythm_key_default = "long_focus"
        else: chosen_rhythm_key_default = "simple"
        
        bass_vol_range = (max(20, int(accompaniment_base_vol_range[0]*0.7)), max(30,int(accompaniment_base_vol_range[1]*0.8)))
        chord_vol_range = (max(20, int(accompaniment_base_vol_range[0]*0.6)), max(30,int(accompaniment_base_vol_range[1]*0.75)))
        
        # Global prog_idx needs to be accessible by _get_triad_notes if it uses it for V7 detection
        # It's better to pass prog_idx or the current roman numeral to _get_triad_notes
        self.prog_idx_global = 0 # Use self if need to access in helper without passing

        key_diatonic_qualities = self.MAJOR_DIATONIC_CHORDS if is_major_mode else self.MINOR_DIATONIC_CHORDS
        scale_intervals_for_chords = [0,2,4,5,7,9,11] if is_major_mode else [0,2,3,5,7,8,10]
        track_event_times = {"melody": 0, "bass": 0, "chords": 0}

        def add_message_to_track(track_obj, track_name_key, msg_type, target_abs_tick, **kwargs):
            delta_t = target_abs_tick - track_event_times[track_name_key]
            if delta_t < 0: delta_t = 0 
            msg_kwargs = kwargs.copy(); msg_kwargs['time'] = int(delta_t) 
            track_obj.append(mido.Message(msg_type, **msg_kwargs))
            track_event_times[track_name_key] = target_abs_tick
        
        def add_note_sequence(track_obj, track_name_key, channel, notes, velocity, start_abs_tick, 
                              rhythmic_duration_ticks, sounding_duration_factor=1.0):
            # For piano legato, sounding_duration might slightly exceed rhythmic_duration
            # Let legato_factor be 0-1 for how much of the slot it fills,
            # And an additional small overlap for >1.0 for true legato simulation
            overlap_ticks = 0
            if channel == 0 and sounding_duration_factor > 0.95: # Melody piano
                # if sounding_duration_factor near 1.0, make it fill the slot
                # if > 1.0, it means overlap
                overlap_factor = sounding_duration_factor # Use the direct factor
                # Calculate overlap: if factor is 1.05, it means 5% overlap
                # The note_off happens *after* the rhythmic slot by a small amount
                effective_sounding_duration = int(rhythmic_duration_ticks * overlap_factor)
            else: # Accompaniment, less critical for overlap
                effective_sounding_duration = int(rhythmic_duration_ticks * min(sounding_duration_factor, 1.0)) # Cap at 1.0 for accompaniment

            effective_sounding_duration = max(self.tpb // 16, effective_sounding_duration)

            for note_pitch in notes: 
                add_message_to_track(track_obj, track_name_key, 'note_on', start_abs_tick, channel=channel, note=note_pitch, velocity=int(velocity))
            for note_pitch in notes:
                add_message_to_track(track_obj, track_name_key, 'note_off', start_abs_tick + effective_sounding_duration, channel=channel, note=note_pitch, velocity=0)

        for cycle_num in range(self.cycles):
            num_measures_per_cycle = random.choice([2,3,4]) if form not in ["etude", "prelude"] else 2
            phrase_tendency = random.choice([-1,0,1]) if musical_mood not in ["sadness","fear","serenity","suspense","pessimism","remorse"] else random.choice([-1,-1,0]) 

            for measure_num_in_cycle in range(num_measures_per_cycle):
                measure_start_abs_tick = current_tick_abs
                ticks_in_measure = beats_per_measure * self.tpb
                
                current_roman_numeral = chord_progression_template[self.prog_idx_global % len(chord_progression_template)]
                chord_type_override = None
                if "aug" in current_roman_numeral.lower() or "+" in current_roman_numeral: chord_type_override = "A"
                elif "dim" in current_roman_numeral.lower() or "o" in current_roman_numeral: chord_type_override = "d"

                root_degree_idx = self._map_roman_to_scale_degree(current_roman_numeral)
                chord_root_pitch = harmony_scale_root_notes[root_degree_idx % len(harmony_scale_root_notes)]
                chord_type = chord_type_override if chord_type_override else key_diatonic_qualities[root_degree_idx % len(key_diatonic_qualities)]
                
                # Pass current_roman_numeral or prog_idx to _get_triad_notes if it needs it for V7 etc.
                # For now, it's not used in _get_triad_notes for V7 detection logic (that part was commented out).
                triad_pitches = self._get_triad_notes(chord_root_pitch, is_major_mode, chord_type, scale_intervals_for_chords, musical_mood)
                self.prog_idx_global +=1
                
                # --- Accompaniment Tracks (Bass & Chords) ---
                bass_vel = self._get_velocity_for_shape(bass_vol_range, (measure_num_in_cycle + 0.5) / num_measures_per_cycle, dynamic_shape_hint)
                chord_vel = self._get_velocity_for_shape(chord_vol_range, (measure_num_in_cycle + 0.5) / num_measures_per_cycle, dynamic_shape_hint)

                if form == "waltz":
                    add_note_sequence(bass_track, "bass", 1, [chord_root_pitch], bass_vel, measure_start_abs_tick, self.tpb, sounding_duration_factor=0.9) # Bass slightly staccato
                    add_note_sequence(chord_track, "chords", 2, triad_pitches, chord_vel, measure_start_abs_tick + self.tpb, self.tpb, sounding_duration_factor=0.8)
                    add_note_sequence(chord_track, "chords", 2, triad_pitches, chord_vel, measure_start_abs_tick + self.tpb*2, self.tpb, sounding_duration_factor=0.8)
                else: # Other forms
                    bass_sounding_factor = 0.95 if musical_mood not in ["serenity", "sadness"] else 1.0 # Longer bass for slow moods
                    add_note_sequence(bass_track, "bass", 1, [chord_root_pitch], bass_vel, measure_start_abs_tick, ticks_in_measure, sounding_duration_factor=bass_sounding_factor) # Bass holds longer

                    chord_legato = 0.95 if musical_mood in ["serenity", "sadness", "awe"] else 0.9 
                    if form == "prelude" and random.random() > 0.3: 
                        # ... (arpeggio logic - simplified, ensure it uses add_note_sequence correctly) ...
                        arp_note_dur = self.tpb // (2 if density > 1 else 1) 
                        current_arp_tick = measure_start_abs_tick
                        safe_triad = [p for p in triad_pitches if 20<p<109] or [60]
                        for note in safe_triad + [safe_triad[1%len(safe_triad)]]:
                             if current_arp_tick >= measure_start_abs_tick + ticks_in_measure: break
                             actual_arp_dur = min(arp_note_dur, (measure_start_abs_tick + ticks_in_measure) - current_arp_tick)
                             if actual_arp_dur <=0 : break
                             add_note_sequence(chord_track, "chords", 2, [note], int(chord_vel*0.8), current_arp_tick, actual_arp_dur, sounding_duration_factor=0.98)
                             current_arp_tick += actual_arp_dur
                    else: 
                        add_note_sequence(chord_track, "chords", 2, triad_pitches, chord_vel, measure_start_abs_tick, ticks_in_measure, sounding_duration_factor=chord_legato)
                
                # --- Melody Track ---
                current_melody_ticks_in_measure = 0
                # Choose rhythm pattern based on mood
                if musical_mood in rhythmic_patterns: chosen_rhythm_key = musical_mood # if specific pattern for mood
                else: chosen_rhythm_key = chosen_rhythm_key_default if chosen_rhythm_key_default in rhythmic_patterns else random.choice(list(rhythmic_patterns.keys()))
                
                _avg_slot_for_melody = avg_rhythmic_slot_duration_ticks # This already includes base_dur_mult
                current_rhythmic_pattern_melody = list(rhythmic_patterns[chosen_rhythm_key])
                if form == "etude": 
                    _avg_slot_for_melody = (self.tpb / max(1.0, density * 1.8)) * base_note_duration_multiplier # Etudes faster but respect base_dur_mult
                    current_rhythmic_pattern_melody = [0.5, 0.5] 
                
                rhythm_idx = 0
                while current_melody_ticks_in_measure < ticks_in_measure:
                    # ... (progress calculations) ...
                    progress_in_measure = current_melody_ticks_in_measure / ticks_in_measure if ticks_in_measure > 0 else 0
                    progress_in_phrase = (measure_num_in_cycle + progress_in_measure) / num_measures_per_cycle if num_measures_per_cycle > 0 else 0

                    duration_factor = current_rhythmic_pattern_melody[rhythm_idx % len(current_rhythmic_pattern_melody)]
                    rhythmic_slot_duration = int(_avg_slot_for_melody * duration_factor)
                    rhythmic_slot_duration = max(self.tpb // 8, rhythmic_slot_duration) # Min 32nd note slot
                    
                    if current_melody_ticks_in_measure + rhythmic_slot_duration > ticks_in_measure:
                        rhythmic_slot_duration = ticks_in_measure - current_melody_ticks_in_measure
                    if rhythmic_slot_duration <= self.tpb // 32 : break 

                    current_melody_note_idx, prev_melody_step = self._select_next_note_index(
                        current_melody_note_idx, melody_scale_len, melodic_range_hint, prev_melody_step, phrase_tendency, musical_mood
                    )
                    note_pitch = melody_scale_notes[current_melody_note_idx]
                    mel_vel = self._get_velocity_for_shape(melody_vol_range, progress_in_phrase, dynamic_shape_hint)
                    
                    add_note_sequence(melody_track, "melody", 0, [note_pitch], mel_vel, 
                                      measure_start_abs_tick + current_melody_ticks_in_measure, 
                                      rhythmic_slot_duration, 
                                      sounding_duration_factor=legato_intensity) # Use legato_intensity
                    
                    current_melody_ticks_in_measure += rhythmic_slot_duration
                    rhythm_idx += 1
                current_tick_abs += ticks_in_measure 
        return mid

    def text_to_midi(self, text, out_path, form_override=None, emotion_override=None):
        # This method needs to unpack 15 values from analyze_text
        # And pass all 15 to _build_multi_track_music in the correct order
        if not text or len(text.strip()) < 10:
            text = "Neutral placeholder text for music generation due to short input."
            print(f"Warning: Input text for {out_path} was too short. Using placeholder.")

        (analyzed_mood_val, analyzed_topic_val, analyzed_form_val, 
         analyzed_melody_scale_val, analyzed_tempo_val, analyzed_density_val, 
         analyzed_melody_specific_vr_val, analyzed_accompaniment_base_vr_val, 
         analyzed_mrh_val, analyzed_prog_val, analyzed_harmony_scale_val, 
         analyzed_is_major_val, analyzed_dynamic_shape_val,
         analyzed_legato_intensity_val, analyzed_base_dur_mult_val # New
         ) = self.analyze_text(text) # Ensure 15 values are returned and unpacked
        
        final_form = form_override or analyzed_form_val
        final_mood  = emotion_override or analyzed_mood_val 
        
        # Initialize final values
        final_tempo = analyzed_tempo_val
        final_density = analyzed_density_val
        final_melody_vr = analyzed_melody_specific_vr_val
        final_accomp_vr = analyzed_accompaniment_base_vr_val
        final_mrh = analyzed_mrh_val
        final_melody_scale = analyzed_melody_scale_val
        final_prog = analyzed_prog_val
        final_harmony_scale = analyzed_harmony_scale_val
        final_is_major = analyzed_is_major_val
        final_dynamic_shape = analyzed_dynamic_shape_val
        final_legato_intensity = analyzed_legato_intensity_val
        final_base_dur_mult = analyzed_base_dur_mult_val

        if emotion_override: # If final_mood is based on override
            mood_params = self.MUSICAL_MOOD_PARAMS.get(final_mood, self.MUSICAL_MOOD_PARAMS["neutral"])
            original_mood_for_recalc = analyzed_mood_val # Mood derived from text
            original_mood_params_for_recalc = self.MUSICAL_MOOD_PARAMS.get(original_mood_for_recalc, self.MUSICAL_MOOD_PARAMS["neutral"])

            read_adj_approx = analyzed_tempo_val - self.base_tempo - original_mood_params_for_recalc["tempo"] 
            final_tempo = max(30, min(180, self.base_tempo + mood_params["tempo"] + read_adj_approx))
            
            vol_lo, vol_hi = mood_params["vol"]
            mel_vol_adj = mood_params.get("mel_adj",0)
            final_melody_vr = (max(10, vol_lo + mel_vol_adj), max(20, vol_hi + mel_vol_adj))
            final_accomp_vr = mood_params["vol"]

            final_mrh = mood_params.get("mel_range", (2,5))
            final_prog = mood_params.get("prog", ["I", "IV", "V", "I"])
            final_dynamic_shape = mood_params.get("dyn_shape", "flat")
            final_legato_intensity = mood_params.get("legato_i", 0.85)
            final_base_dur_mult = mood_params.get("base_dur_mult", 1.0)
            
            if "is_major" in mood_params: final_is_major = mood_params["is_major"]
            else: final_is_major = final_mood in ("joy","surprise","love","neutral","serenity","trust","awe","optimism")
            
            original_base_density_approx = analyzed_density_val / (0.5 + original_mood_params_for_recalc.get("r_complex", 0.5) + 1e-6) # Adjusted denominator
            new_r_complex_factor = mood_params.get("r_complex", 0.5)
            final_density = np.clip(original_base_density_approx * (0.5 + new_r_complex_factor), 0.15, 4.0)

            new_mode_intervals = [0,2,4,5,7,9,11] if final_is_major else [0,2,3,5,7,8,10]
            original_is_major_for_shift = self.MUSICAL_MOOD_PARAMS.get(original_mood_for_recalc,{}).get("is_major", original_mood_for_recalc in ("joy","surprise","love","neutral","serenity","trust","awe","optimism"))
            original_mode_intervals_for_shift = [0,2,4,5,7,9,11] if original_is_major_for_shift else [0,2,3,5,7,8,10]
            
            if analyzed_melody_scale_val and original_mode_intervals_for_shift:
                original_shift_val = (analyzed_melody_scale_val[0] - 60 - original_mode_intervals_for_shift[0]) % 12
            else: original_shift_val = 0 
            
            final_melody_scale = sorted([60 + i + original_shift_val for i in new_mode_intervals] + [72 + i + original_shift_val for i in new_mode_intervals])
            harmony_base_octave = 48
            final_harmony_scale = sorted([harmony_base_octave + i + original_shift_val for i in new_mode_intervals])

        print(f"Generating Multi-Track V4: Mood={final_mood}, Form={final_form}, Tempo={final_tempo}, Legato={final_legato_intensity:.2f}, BaseDurM={final_base_dur_mult:.2f}")

        midi_obj = self._build_multi_track_music(
            final_mood, analyzed_topic_val, final_form, final_melody_scale, final_tempo, 
            final_density, final_melody_vr, final_accomp_vr, final_mrh, 
            final_prog, final_harmony_scale, final_is_major,
            final_dynamic_shape, final_legato_intensity, final_base_dur_mult # Pass all 15
        )
        
        midi_obj.save(out_path)
        print(f"▶️ Saved Multi-Track V4: {out_path} [{final_mood}/{analyzed_topic_val}/{final_form} @ {final_tempo} BPM]")


# --- extract_feats and rule_emotion functions as before ---
# ... (copy from your working version) ...
def extract_feats(path):
    try:
        mid = mido.MidiFile(str(path), clip=True)
    except Exception: 
        return None
    notes, vels, tempos_val = [], [], []
    tpb = mid.ticks_per_beat if mid.ticks_per_beat > 0 else 480 
    ticks_total=0
    first_tempo_us = None

    for tr_idx, tr in enumerate(mid.tracks):
        current_time_in_track = 0
        for msg_idx, msg in enumerate(tr):
            current_time_in_track += msg.time
            if msg.type=="set_tempo": 
                tempos_val.append(msg.tempo)
                if first_tempo_us is None: 
                    first_tempo_us = msg.tempo
            elif msg.type=="note_on" and msg.velocity>0:
                notes.append(msg.note); vels.append(msg.velocity)
        ticks_total = max(ticks_total, current_time_in_track)

    if not tempos_val and first_tempo_us is None: 
        tempo_us = 500000 
    elif first_tempo_us is not None:
        tempo_us = first_tempo_us
    else: 
        tempo_us = tempos_val[0]

    bpm = mido.tempo2bpm(tempo_us) if tempo_us > 0 else 120
    dyn_mean = float(np.mean(vels)) if vels else 64.0
    secs = mido.tick2second(ticks_total, tpb, tempo_us) if tpb > 0 and tempo_us > 0 else 0
    density_feat  = len(notes)/secs if secs > 0 else 0.0 
    prange   = (max(notes)-min(notes)) if len(notes) > 1 else 0 
    return np.array([bpm,dyn_mean,density_feat,prange])

def rule_emotion(feats):
    if feats is None: return "neutral"
    bpm,dyn,dens,pr=feats
    # These labels should ideally match keys in MUSICAL_MOOD_PARAMS or be mapped by _derive_musical_mood
    if bpm>125 and dyn>65 and dens > 2.5: return "joy" 
    if bpm<85 and dens < 2.5 and dyn < 60: return "sadness" 
    if dens<2.0 and bpm < 105 and dyn < 70 : return "neutral" # Was "calm", map to neutral or serenity
    if bpm > 110 and dyn > 60 and dens > 2.0 : return "joy" # Was "energetic", map to joy or anticipation
    return "neutral"

# %%
# Cell 2: clean folders + classify Maestro forms & LMD emotions
import shutil, os, glob
from pathlib import Path
import pandas as pd
import random

# 1) clear old splits
for d in ["data/maestro_subset","data/lmd_emotion"]:
    if os.path.isdir(d): shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

# 2) Maestro → forms (via CSV)
CSV = Path("data/maestro/maestro-v3.0.0/maestro-v3.0.0.csv")
MAE = Path("data/maestro/maestro-v3.0.0")
OUT = Path("data/maestro_subset")
FORMS = ["sonata","etude","prelude","waltz","nocturne"]
df = pd.read_csv(CSV)
for _,r in df.iterrows():
    f = next((x for x in FORMS if x in str(r.canonical_title).lower()), None)
    if not f: continue
    src = MAE / r.midi_filename
    if src.exists():
        dst = OUT/f/src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src,dst)

# 3) LMD → emotions (fast RF on a 500‐file subset)
all_mid = glob.glob("data/lmd_clean/**/*.mid", recursive=True)
subset = random.sample(all_mid, min(500,len(all_mid)))
X,y = [],[]
for fn in subset:
    feats = extract_feats(fn)
    if feats is None: continue
    X.append(feats); y.append(rule_emotion(feats))
X = np.stack(X)
le = LabelEncoder().fit(y)
y_enc = le.transform(y)
clf = RandomForestClassifier(n_estimators=20, random_state=0)
print("CV emo‐acc:", np.mean(cross_val_score(clf,X,y_enc,cv=3)))
clf.fit(X,y_enc)
os.makedirs("models",exist_ok=True)
pickle.dump((le,clf), open("models/lmd_emotion_clf.pkl","wb"))

# 4) apply RF to all LMD MIDI
OUT2 = Path("data/lmd_emotion")
for fn in glob.glob("data/lmd_clean/**/*.mid", recursive=True):
    feats = extract_feats(fn)
    if feats is None: continue
    emo = le.inverse_transform([clf.predict(feats.reshape(1,-1))[0]])[0]
    dst = OUT2/emo/Path(fn).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(fn,dst)

print("✅ Maestro & LMD re‐classified fresh")


# %%
# %%
# Cell 3: chapter/scene → form+emotion → background MIDI
import re
from pathlib import Path
import shutil # For rmtree
import os     # For path operations and chmod
import stat   # For chmod constants

# --- Define the main output directory ---
MAIN_OUTPUT_DIR = Path("background_music_nb")

# --- Helper function to remove directory (handles read-only files) ---
# This is useful if previous runs left files that are hard to delete.
def _clear_directory_robustly(dir_path):
    def onerror(func, p, exc_info):
        # If a permission error occurs, try to change permissions and retry
        if isinstance(exc_info[1], PermissionError):
            try:
                os.chmod(p, stat.S_IWRITE) # Grant write permission
                func(p) # Retry the operation (e.g., os.remove or os.rmdir)
            except Exception as e:
                print(f"    Still failed to remove {p} after chmod: {e}")
        else:
            print(f"    Error during rmtree for {p}: {exc_info[1]}")

    if dir_path.is_dir():
        print(f"Clearing old output directory: {dir_path}")
        try:
            shutil.rmtree(dir_path, onerror=onerror)
        except Exception as e: # Catch final rmtree errors if onerror didn't solve all
            print(f"  Could not completely remove {dir_path}: {e}. Trying to create anyway.")
    elif dir_path.exists(): # It's a file, not a directory
        print(f"Found a file at {dir_path} instead of a directory. Removing it.")
        try:
            os.remove(dir_path)
        except Exception as e:
            print(f"  Could not remove file {dir_path}: {e}")


# --- Clear the main output directory before starting ---
_clear_directory_robustly(MAIN_OUTPUT_DIR)

# --- Ensure the main output directory exists (it will be recreated) ---
MAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # exist_ok=True is fine here as we just cleared it

# --- Helper functions for text segmentation (roman_to_int, parse_act_scene_heading, split_into_segments) ---
# (These functions remain the same as your last version)
def roman_to_int(s):
    if not s: return 0
    s = s.upper()
    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    i = 0; num = 0
    while i < len(s):
        s1 = roman_map.get(s[i], 0)
        if (i + 1) < len(s):
            s2 = roman_map.get(s[i + 1], 0)
            if s1 >= s2: num = num + s1; i = i + 1
            else: num = num + s2 - s1; i = i + 2
        else: num = num + s1; i = i + 1
    return num if num > 0 else None

def parse_act_scene_heading(heading_text):
    act_num_str, scene_num_str = None, None
    act_scene_match = re.search(r"ACT\s+(?P<act>[IVXLCDM\d]+)[\.,\s]*SCENE\s+(?P<scene>[IVXLCDM\d]+)", heading_text, re.IGNORECASE)
    if act_scene_match:
        act_str = act_scene_match.group("act").strip().upper(); scene_str = act_scene_match.group("scene").strip().upper()
        act_num = roman_to_int(act_str) if not act_str.isdigit() else int(act_str)
        scene_num = roman_to_int(scene_str) if not scene_str.isdigit() else int(scene_str)
        if act_num: act_num_str = f"Act{act_num}"
        if scene_num: scene_num_str = f"Scene{scene_num}"
        return act_num_str, scene_num_str
    act_match = re.search(r"^\s*ACT\s+(?P<act>[IVXLCDM\d]+)", heading_text, re.IGNORECASE | re.MULTILINE)
    if act_match:
        act_str = act_match.group("act").strip().upper()
        act_num = roman_to_int(act_str) if not act_str.isdigit() else int(act_str)
        if act_num: act_num_str = f"Act{act_num}"
    scene_match = re.search(r"^\s*SCENE\s+(?P<scene>[IVXLCDM\d]+)", heading_text, re.IGNORECASE | re.MULTILINE)
    if scene_match:
        scene_str = scene_match.group("scene").strip().upper()
        scene_num = roman_to_int(scene_str) if not scene_str.isdigit() else int(scene_str)
        if scene_num: scene_num_str = f"Scene{scene_num}"
    return act_num_str, scene_num_str

def split_into_segments(text_content):
    segments = []
    chapter_pattern = re.compile(r"(Chapter\s+([IVXLCDM\d\w]+(?:[\s-][IVXLCDM\d\w]+)*)[\.\s\n]*)([\s\S]*?)(?=(?:Chapter\s+[IVXLCDM\d\w])|\Z)", re.IGNORECASE)
    chapter_matches = list(chapter_pattern.finditer(text_content))
    if chapter_matches:
        print("Detected Chapter structure.")
        for i, match in enumerate(chapter_matches):
            chapter_num_text = match.group(2).strip()
            segment_text = match.group(3).strip()
            identifier = f"Chap{chapter_num_text.replace(' ', '_')}"
            segments.append((identifier, segment_text))
        return segments
    print("No chapters found, trying Act/Scene structure.")
    potential_headings = []
    for match in re.finditer(r"^(ACT\s+[IVXLCDM\d]+(?:[\.,\s]*SCENE\s+[IVXLCDM\d]+)?|SCENE\s+[IVXLCDM\d]+)", text_content, re.IGNORECASE | re.MULTILINE):
        potential_headings.append({'text': match.group(0).strip(), 'start': match.start(), 'end': match.end()})
    if not potential_headings:
        print("No Act/Scene structure found, treating as single segment.")
        return [("Segment1", text_content.strip())]
    potential_headings.sort(key=lambda x: x['start'])
    current_act_str = "Act0"
    for i, heading_info in enumerate(potential_headings):
        heading_text = heading_info['text']
        segment_start = heading_info['end']
        segment_end = potential_headings[i+1]['start'] if (i + 1) < len(potential_headings) else len(text_content)
        segment_text = text_content[segment_start:segment_end].strip()
        act_str, scene_str = parse_act_scene_heading(heading_text)
        if act_str: current_act_str = act_str
        if scene_str: identifier = f"{current_act_str}_{scene_str}"
        elif act_str and not scene_str:
            identifier = f"{act_str}_Intro"
            if len(segment_text) < 100 and ( (i + 1) < len(potential_headings) and "SCENE" in potential_headings[i+1]['text'].upper() ):
                 continue 
        else: identifier = f"UnknownSegment{i+1}"
        if segment_text: segments.append((identifier, segment_text))
    if not segments and text_content: return [("CompleteText", text_content.strip())]
    return segments

# --- Ensure 'gen' is instantiated (should be from Cell 1 after kernel restart and running Cell 1) ---
if 'gen' not in locals() or 'gen' not in globals():
    print("Instantiating FormAwareTextToMusic in Cell 3 as 'gen' was not found...")
    # This assumes FormAwareTextToMusic class is defined (i.e., Cell 1 has been run)
    gen = FormAwareTextToMusic() 
else:
    print("'gen' object found. Proceeding.")


# --- Main Loop to Process Books and Generate MIDI ---
for book_path in Path("books").glob("*.txt"):
    print(f"\nProcessing book: {book_path.name}")
    text_content = book_path.read_text(encoding="utf-8")
    
    segments = split_into_segments(text_content)

    if not segments:
        print(f"Could not segment book: {book_path.name}. Skipping.")
        continue

    # Create the book-specific subdirectory *inside* the (now fresh) MAIN_OUTPUT_DIR
    book_output_dir = MAIN_OUTPUT_DIR / book_path.stem
    book_output_dir.mkdir(parents=True, exist_ok=True) # exist_ok is fine here

    for seg_idx, (segment_id, segment_text) in enumerate(segments):
        print(f"  Generating for: {segment_id}")
        if not segment_text.strip():
            print(f"    Skipping empty segment: {segment_id}")
            continue

        try:
            # Unpack all 15 values from analyze_text
            (analyzed_mood, analyzed_topic, analyzed_form, 
             analyzed_melody_scale, analyzed_tempo, analyzed_density, 
             analyzed_melody_vr, analyzed_accomp_vr, analyzed_mrh, 
             analyzed_prog_template, analyzed_harmony_roots, 
             analyzed_is_major, analyzed_dynamic_shape, 
             analyzed_legato_intensity, analyzed_base_dur_mult
             ) = gen.analyze_text(segment_text)
        except Exception as e:
            print(f"    Error analyzing text for {segment_id}: {e}")
            # Optionally, provide default parameters to still attempt MIDI generation
            # analyzed_mood, analyzed_form = "neutral", "prelude" # etc.
            continue # Skip this segment if analysis fails
        
        safe_segment_id = re.sub(r'[^\w\.-]', '_', segment_id)
        outm_filename = book_output_dir / f"{book_path.stem}_{safe_segment_id}_{analyzed_form}_{analyzed_mood}.mid"
        
        try:
            gen.text_to_midi(segment_text, str(outm_filename), 
                             form_override=analyzed_form, 
                             emotion_override=analyzed_mood)
            print(f"    → {outm_filename.name}")
        except Exception as e:
            print(f"    Error generating MIDI for {segment_id}: {e}")

print("\n✅ All text segments processed.")

# %%



