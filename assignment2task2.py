# %%
!pip install tf-keras

# %%
# %%
# Cell 1: Pipeline class + MIDI‐emotion helpers (standalone)
# (MULTI-TRACK V2 - Enhanced Dynamics, Smoother Melody, Clearer Channels)

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
        # ... (same __init__ as your last version) ...
        self.emoc     = pipeline("text-classification",
                                 model="j-hartmann/emotion-english-distilroberta-base",
                                 return_all_scores=True)
        self.zs_topic = pipeline("zero-shot-classification",
                                 model="facebook/bart-large-mnli")
        self.topics   = ["love","betrayal","death","fate","family","violence"]
        self.zs_form  = pipeline("zero-shot-classification",
                                 model="facebook/bart-large-mnli")
        self.forms    = ["sonata","etude","prelude","waltz","nocturne","fugue"]
        self.tfidf    = TfidfVectorizer(max_features=5, stop_words="english", min_df=1)
        self.nlp      = spacy.load("en_core_web_sm")
        self.base_tempo = base_tempo
        self.tpb        = tpb 
        self.cycles     = cycles

        self.EMO_PARAMS = { 
          "joy":     {"tempo":+20, "vol":(75,105), "melodic_range": (2, 5), "rhythmic_complexity": 0.7, "progression": ["I", "V", "vi", "IV"], "dynamic_shape": "crescendo"},
          "sadness": {"tempo":-20, "vol":(35,65),  "melodic_range": (1, 3), "rhythmic_complexity": 0.3, "progression": ["vi", "IV", "I", "V"], "dynamic_shape": "diminuendo"},
          "anger":   {"tempo":+10, "vol":(80,115), "melodic_range": (2, 6), "rhythmic_complexity": 0.6, "progression": ["i", "VI", "III", "VII"], "dynamic_shape": "accented"}, # Using minor context for anger
          "fear":    {"tempo":  0, "vol":(45,75),  "melodic_range": (1, 4), "rhythmic_complexity": 0.4, "progression": ["vi", "iidim", "V", "i"], "dynamic_shape": "subdued_var"},
          "surprise":{"tempo":+15,"vol":(70,100), "melodic_range": (3, 7), "rhythmic_complexity": 0.8, "progression": ["I", "bVI", "IV", "V"], "dynamic_shape": "sudden_loud"}, # bVI for surprise
          "love":    {"tempo":+10, "vol":(65,95),  "melodic_range": (2, 4), "rhythmic_complexity": 0.5, "progression": ["I", "vi", "ii", "V"], "dynamic_shape": "gentle_swell"},
          "neutral": {"tempo":  0, "vol":(55,80),  "melodic_range": (2, 5), "rhythmic_complexity": 0.5, "progression": ["I", "IV", "V", "ii"], "dynamic_shape": "flat"},
        }
        self.TOPIC_PARAMS = {
          "love":1.2,"betrayal":0.8,"death":0.6,
          "fate":1.0,"family":1.1,"violence":0.7,"neutral":1.0
        }
        self.MAJOR_DIATONIC_CHORDS = ["M", "m", "m", "M", "M", "m", "d"]
        self.MINOR_DIATONIC_CHORDS = ["m", "d", "M", "m", "m", "M", "M"]


    def analyze_text(self, text):
        # ... (same analyze_text as your last version, ensure it returns 'dynamic_shape') ...
        emos = self.emoc(text[:512])[0]
        emo  = max(emos, key=lambda x:x["score"])["label"].lower()
        eparam = self.EMO_PARAMS.get(emo, self.EMO_PARAMS["neutral"])
        
        top   = self.zs_topic(text[:512], candidate_labels=self.topics)["labels"][0].lower()
        tmult = self.TOPIC_PARAMS.get(top,1.0)
        form_label  = self.zs_form(text[:512], candidate_labels=self.forms)["labels"][0].lower()
        
        try:
            self.tfidf.fit([text])
            kws = list(self.tfidf.get_feature_names_out())
        except: kws = []
        
        doc       = self.nlp(text[:5000]) 
        ne_ratio  = len(doc.ents)/max(len(doc),1) if doc and len(doc) > 0 else 0
        quotes    = len(re.findall(r'“[^”]+”', text))
        num_sents = len(list(doc.sents)) if doc else 0
        dlg_ratio = quotes/max(num_sents,1) if num_sents > 0 else 0
        
        read     = textstat.flesch_reading_ease(text)
        read_adj = int((read-60)/10)
        
        tempo    = max(40, min(200, self.base_tempo + eparam["tempo"] + read_adj))
        base_density = tmult * (1 + dlg_ratio) 
        rhythmic_complexity_factor = eparam.get("rhythmic_complexity", 0.5)
        density = np.clip(base_density * (1 + rhythmic_complexity_factor), 0.5, 4.0)

        vol_lo, vol_hi = eparam["vol"]
        melodic_range_hint = eparam.get("melodic_range", (2,5)) 
        chord_progression_template = eparam.get("progression", ["I", "IV", "V", "I"])
        dynamic_shape_hint = eparam.get("dynamic_shape", "flat")


        # Corrected is_major_mode for anger to reflect progression using minor
        is_major_mode = emo in ("joy","surprise","love", "neutral") # Neutral can be major
        if emo == "anger": is_major_mode = False # Make anger use minor mode progressions more consistently

        mode_intervals = [0,2,4,5,7,9,11] if is_major_mode else [0,2,3,5,7,8,10]
        shift = (sum(hash(k) for k in kws)+int(ne_ratio*12)) % 12 
        scale = sorted([60 + i + shift for i in mode_intervals] + [72 + i + shift for i in mode_intervals])
        harmony_base_octave = 48 
        harmony_scale_root_notes = sorted([harmony_base_octave + i + shift for i in mode_intervals])


        return (emo, top, form_label, scale, tempo, density, (vol_lo, vol_hi), melodic_range_hint,
                chord_progression_template, harmony_scale_root_notes, is_major_mode, dynamic_shape_hint)

    def _get_velocity_for_shape(self, base_vol_range, progress_in_phrase, dynamic_shape_hint):
        """ Generates velocity based on dynamic shape over a phrase (progress 0 to 1) """
        lo, hi = base_vol_range
        span = hi - lo
        
        if dynamic_shape_hint == "crescendo":
            return int(lo + span * progress_in_phrase)
        elif dynamic_shape_hint == "diminuendo":
            return int(hi - span * progress_in_phrase)
        elif dynamic_shape_hint == "gentle_swell": # Swell in the middle
            return int(lo + span * (0.5 + 0.5 * np.sin((progress_in_phrase - 0.5) * np.pi)))
        elif dynamic_shape_hint == "accented": # Louder at start, then softer
            return int(hi - span * progress_in_phrase * 0.7) if progress_in_phrase > 0.1 else hi
        elif dynamic_shape_hint == "sudden_loud": # Mostly soft, then loud spike
            return hi if progress_in_phrase > 0.8 else int(lo + span * 0.2)
        elif dynamic_shape_hint == "subdued_var": # Mostly soft with slight variation
            return int(lo + span * (0.1 + 0.2 * np.sin(progress_in_phrase * 4 * np.pi)))
        else: # flat or other
            return random.randint(lo, hi)


    def _select_next_note_index(self, current_note_idx, scale_len, melodic_range_hint, prev_step=0, tendency=0):
        """
        Selects the next note index.
        prev_step: the previous interval step taken.
        tendency: +1 for upward, -1 for downward, 0 for neutral.
        """
        min_jump, max_jump = melodic_range_hint
        
        # Prioritize stepwise motion, then small jumps
        possible_intervals = [-2, -1, 1, 2] # More stepwise
        # Add slightly larger jumps with lower probability
        for j in range(3, max_jump + 1):
            if random.random() < 0.4 / j : possible_intervals.extend([-j, j])

        weights = []
        for interval in possible_intervals:
            w = 1.0 / (abs(interval) + 0.1) # Favor smaller intervals
            if abs(interval) > min_jump: w *= 0.5 # Penalize jumps larger than min_jump
            if interval == -prev_step and abs(prev_step) > 1: w*= 0.3 # Penalize immediate reversal of a jump
            if tendency != 0 and np.sign(interval) == tendency: w *= 1.5 # Follow tendency
            weights.append(w)
        
        if not possible_intervals: # Should not happen if scale_len > 1
             return (current_note_idx + 1) % scale_len, 1

        chosen_step = random.choices(possible_intervals, weights=weights, k=1)[0]
        next_idx = current_note_idx + chosen_step
        next_idx = np.clip(next_idx, 0, scale_len - 1)
        
        # If stuck or step too small, try a slightly larger random step
        if next_idx == current_note_idx and scale_len > 1:
            chosen_step = random.choice([-1,1]) if scale_len > 1 else 0
            next_idx = (current_note_idx + chosen_step + scale_len) % scale_len
            
        return int(next_idx), chosen_step


    def _get_triad_notes(self, root_note_pitch, is_major_key, chord_type_str, scale_intervals):
        # ... (same as before) ...
        third_interval = 4 if chord_type_str == "M" else 3 
        if chord_type_str == "d": 
            fifth_interval = 6 
        else: 
            fifth_interval = 7 
        
        n1 = root_note_pitch
        n2 = root_note_pitch + third_interval
        n3 = root_note_pitch + fifth_interval
        return [n1, n2, n3]


    def _map_roman_to_scale_degree(self, roman_numeral_str):
        # ... (same as before) ...
        mapping = {"i":0, "I":0, "ii":1, "II":1, "iii":2, "III":2, "iv":3, "IV":3, 
                   "v":4, "V":4, "vi":5, "VI":5, "vii":6, "VII":6}
        return mapping.get(roman_numeral_str.replace("dim","").replace("o",""), 0)


    def _build_multi_track_music(self, emo, topic, form, melody_scale, tempo, density, 
                                 main_vol_range, melodic_range_hint, chord_progression_template,
                                 harmony_scale_root_notes, is_major_mode, dynamic_shape_hint):
        mid = mido.MidiFile(ticks_per_beat=self.tpb)
        
        melody_track = mido.MidiTrack(); melody_track.name = "Melody (Ch 0)"; mid.tracks.append(melody_track)
        bass_track = mido.MidiTrack(); bass_track.name = "Bass (Ch 1)"; mid.tracks.append(bass_track)
        chord_track = mido.MidiTrack(); chord_track.name = "Chords (Ch 2)"; mid.tracks.append(chord_track)

        melody_track.append(mido.Message('program_change', channel=0, program=0, time=0))   
        bass_track.append(mido.Message('program_change', channel=1, program=33, time=0))    
        chord_track.append(mido.Message('program_change', channel=2, program=48, time=0))   

        is_waltz = (form == "waltz")
        beats_per_measure = 3 if is_waltz else 4
        
        current_tick_abs = 0 
        melody_track.append(mido.MetaMessage('time_signature', numerator=beats_per_measure, denominator=4, time=0))
        melody_track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))

        if not melody_scale: melody_scale = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83] 
        melody_scale_len = len(melody_scale)
        current_melody_note_idx = random.randint(melody_scale_len // 4, melody_scale_len // 2) 
        prev_melody_step = 0 

        avg_melody_note_duration_ticks = self.tpb / density
        rhythmic_patterns = {
            "simple": [1.0], "varied1": [1.5, 0.5], "varied2": [0.75, 0.25, 1.0],"dotted": [0.75,0.25],
            "even": [0.5, 0.5] if density >=1.5 else ([1.0] if density >=0.75 else [2.0]) 
        }
        
        bass_vol_range = (max(20, int(main_vol_range[0]*0.5)), max(30,int(main_vol_range[1]*0.65)))
        chord_vol_range = (max(20, int(main_vol_range[0]*0.4)), max(30,int(main_vol_range[1]*0.6)))

        prog_idx = 0 
        key_diatonic_qualities = self.MAJOR_DIATONIC_CHORDS if is_major_mode else self.MINOR_DIATONIC_CHORDS
        scale_intervals_for_chords = [0,2,4,5,7,9,11] if is_major_mode else [0,2,3,5,7,8,10]
        
        # CORRECTED: Use string keys for track_event_times
        track_event_times = {"melody": 0, "bass": 0, "chords": 0}

        # Helper functions now take track_obj (the MidiTrack instance) AND track_name_key (the string key)
        def add_message_to_track(track_obj, track_name_key, msg_type, target_abs_tick, **kwargs):
            delta_t = target_abs_tick - track_event_times[track_name_key]
            if delta_t < 0: delta_t = 0 
            msg_kwargs = kwargs.copy()
            msg_kwargs['time'] = int(delta_t) # Ensure delta_t is int for mido
            track_obj.append(mido.Message(msg_type, **msg_kwargs))
            track_event_times[track_name_key] = target_abs_tick # Store the absolute tick
        
        def add_note_sequence(track_obj, track_name_key, channel, notes, velocity, start_abs_tick, duration_abs_ticks):
            for note_pitch in notes: 
                add_message_to_track(track_obj, track_name_key, 'note_on', start_abs_tick, channel=channel, note=note_pitch, velocity=int(velocity))
            # Turn off all notes at the end of duration
            for note_pitch in notes:
                add_message_to_track(track_obj, track_name_key, 'note_off', start_abs_tick + duration_abs_ticks, channel=channel, note=note_pitch, velocity=0)


        for cycle_num in range(self.cycles):
            num_measures_per_cycle = random.choice([2,3,4]) if form not in ["etude", "prelude"] else 2
            phrase_tendency = random.choice([-1, 0, 1]) if emo not in ["sadness", "fear"] else random.choice([-1, -1, 0]) 

            for measure_num_in_cycle in range(num_measures_per_cycle):
                measure_start_abs_tick = current_tick_abs
                ticks_in_measure = beats_per_measure * self.tpb
                
                roman_numeral = chord_progression_template[prog_idx % len(chord_progression_template)]
                root_degree_idx = self._map_roman_to_scale_degree(roman_numeral)
                chord_root_pitch = harmony_scale_root_notes[root_degree_idx % len(harmony_scale_root_notes)]
                chord_type = key_diatonic_qualities[root_degree_idx % len(key_diatonic_qualities)]
                if roman_numeral.lower() == "iidim" or (emo == "fear" and root_degree_idx == 1 and not is_major_mode): chord_type = "d"
                triad_pitches = self._get_triad_notes(chord_root_pitch, is_major_mode, chord_type, scale_intervals_for_chords)
                prog_idx +=1

                bass_vel = self._get_velocity_for_shape(bass_vol_range, (measure_num_in_cycle + 0.5) / num_measures_per_cycle, dynamic_shape_hint)
                if form == "waltz":
                    add_note_sequence(bass_track, "bass", 1, [chord_root_pitch], bass_vel, measure_start_abs_tick, self.tpb)
                else:
                    add_note_sequence(bass_track, "bass", 1, [chord_root_pitch], bass_vel, measure_start_abs_tick, self.tpb * 2) 
                    if beats_per_measure == 4: 
                        alt_bass_note = chord_root_pitch + 7 if random.random() > 0.6 else chord_root_pitch 
                        add_note_sequence(bass_track, "bass", 1, [alt_bass_note], int(bass_vel*0.8), measure_start_abs_tick + self.tpb*2, self.tpb*2)

                chord_vel = self._get_velocity_for_shape(chord_vol_range, (measure_num_in_cycle + 0.5) / num_measures_per_cycle, dynamic_shape_hint)
                if form == "waltz": 
                    add_note_sequence(chord_track, "chords", 2, triad_pitches, chord_vel, measure_start_abs_tick + self.tpb, self.tpb)
                    add_note_sequence(chord_track, "chords", 2, triad_pitches, chord_vel, measure_start_abs_tick + self.tpb*2, self.tpb)
                else: 
                    if form == "prelude" and random.random() > 0.3: 
                        arp_note_dur = self.tpb // (2 if density > 1 else 1) 
                        arp_vel = chord_vel
                        # Make sure arpeggio notes are within reasonable MIDI range
                        safe_triad_pitches_arp = [p for p in triad_pitches if 20 < p < 109]
                        if not safe_triad_pitches_arp: safe_triad_pitches_arp = [60] # fallback
                        
                        current_arp_tick = measure_start_abs_tick
                        arp_notes_to_play = safe_triad_pitches_arp + [safe_triad_pitches_arp[1 % len(safe_triad_pitches_arp)]] # Up and down one note

                        for i, note in enumerate(arp_notes_to_play): 
                            if current_arp_tick >= measure_start_abs_tick + ticks_in_measure: break
                            actual_arp_dur = min(arp_note_dur, (measure_start_abs_tick + ticks_in_measure) - current_arp_tick)
                            if actual_arp_dur <=0: break
                            add_note_sequence(chord_track, "chords", 2, [note], arp_vel, current_arp_tick, actual_arp_dur)
                            current_arp_tick += actual_arp_dur
                            arp_vel = max(20, int(arp_vel * 0.95)) 
                    else: 
                        add_note_sequence(chord_track, "chords", 2, triad_pitches, chord_vel, measure_start_abs_tick, ticks_in_measure)
                
                current_melody_ticks_in_measure = 0
                chosen_rhythm_key = random.choice(list(rhythmic_patterns.keys())) 
                
                _avg_ticks_for_melody = avg_melody_note_duration_ticks
                current_rhythmic_pattern_melody = list(rhythmic_patterns[chosen_rhythm_key])
                if form == "etude": 
                    _avg_ticks_for_melody = self.tpb / max(1.0, density * 1.8) 
                    current_rhythmic_pattern_melody = [0.5, 0.5] 

                rhythm_idx = 0
                while current_melody_ticks_in_measure < ticks_in_measure:
                    progress_in_measure = current_melody_ticks_in_measure / ticks_in_measure if ticks_in_measure > 0 else 0
                    progress_in_phrase = (measure_num_in_cycle + progress_in_measure) / num_measures_per_cycle if num_measures_per_cycle > 0 else 0
                    
                    duration_factor = current_rhythmic_pattern_melody[rhythm_idx % len(current_rhythmic_pattern_melody)]
                    note_duration = int(_avg_ticks_for_melody * duration_factor)
                    note_duration = max(self.tpb // 4, note_duration) 
                    
                    if current_melody_ticks_in_measure + note_duration > ticks_in_measure:
                        note_duration = ticks_in_measure - current_melody_ticks_in_measure
                    if note_duration <= self.tpb // 16: break 

                    current_melody_note_idx, prev_melody_step = self._select_next_note_index(
                        current_melody_note_idx, melody_scale_len, melodic_range_hint, prev_melody_step, phrase_tendency
                    )
                    note_pitch = melody_scale[current_melody_note_idx]
                    mel_vel = self._get_velocity_for_shape(main_vol_range, progress_in_phrase, dynamic_shape_hint)
                    
                    add_note_sequence(melody_track, "melody", 0, [note_pitch], mel_vel, 
                                      measure_start_abs_tick + current_melody_ticks_in_measure, note_duration)
                    
                    current_melody_ticks_in_measure += note_duration
                    rhythm_idx += 1
                
                current_tick_abs += ticks_in_measure 

        return mid

    def text_to_midi(self, text, out_path, form_override=None, emotion_override=None):
        # ... (ensure it unpacks all 8 params from analyze_text, including dynamic_shape_hint) ...
        # ... (and passes all necessary params to _build_multi_track_music) ...
        if not text or len(text.strip()) < 10:
            text = "Neutral placeholder text for music generation due to short input."
            print(f"Warning: Input text for {out_path} was too short. Using placeholder.")

        (analyzed_emo_val, analyzed_topic_val, analyzed_form_val, analyzed_melody_scale_val, 
         analyzed_tempo_val, analyzed_density_val, analyzed_vr_val, analyzed_mrh_val,
         analyzed_prog_val, analyzed_harmony_scale_val, analyzed_is_major_val, 
         analyzed_dynamic_shape_val) = self.analyze_text(text) # Ensure this matches return
        
        final_form = form_override or analyzed_form_val
        final_emo  = emotion_override or analyzed_emo_val
        
        final_tempo = analyzed_tempo_val
        final_density = analyzed_density_val
        final_vr = analyzed_vr_val
        final_mrh = analyzed_mrh_val
        final_melody_scale = analyzed_melody_scale_val
        final_prog = analyzed_prog_val
        final_harmony_scale = analyzed_harmony_scale_val
        final_is_major = analyzed_is_major_val
        final_dynamic_shape = analyzed_dynamic_shape_val


        if emotion_override: 
            eparam = self.EMO_PARAMS.get(final_emo, self.EMO_PARAMS["neutral"])
            original_emo_an = analyzed_emo_val 
            original_eparam_an = self.EMO_PARAMS.get(original_emo_an, self.EMO_PARAMS["neutral"])

            read_adj = analyzed_tempo_val - self.base_tempo - original_eparam_an["tempo"] 
            final_tempo = max(40, min(200, self.base_tempo + eparam["tempo"] + read_adj))
            final_vr = eparam["vol"]
            final_mrh = eparam.get("melodic_range", (2,5))
            final_prog = eparam.get("progression", ["I", "IV", "V", "I"])
            final_dynamic_shape = eparam.get("dynamic_shape", "flat") # Add this
            final_is_major = final_emo in ("joy","surprise","love", "neutral")
            if final_emo == "anger": final_is_major = False


            original_base_density_approx = analyzed_density_val / (1 + original_eparam_an.get("rhythmic_complexity", 0.5))
            new_rhythmic_complexity_factor = eparam.get("rhythmic_complexity", 0.5)
            final_density = np.clip(original_base_density_approx * (1 + new_rhythmic_complexity_factor), 0.5, 4.0)

            new_mode_intervals = [0,2,4,5,7,9,11] if final_is_major else [0,2,3,5,7,8,10]
            original_mode_intervals_an = [0,2,4,5,7,9,11] if analyzed_emo_val in ("joy","surprise","love", "neutral") else [0,2,3,5,7,8,10]
            if analyzed_emo_val == "anger": original_mode_intervals_an = [0,2,3,5,7,8,10]
            
            original_shift_val = (analyzed_melody_scale_val[0] - 60 - original_mode_intervals_an[0]) % 12
            
            final_melody_scale = sorted([60 + i + original_shift_val for i in new_mode_intervals] + [72 + i + original_shift_val for i in new_mode_intervals])
            harmony_base_octave = 48
            final_harmony_scale = sorted([harmony_base_octave + i + original_shift_val for i in new_mode_intervals])


        print(f"Generating Multi-Track V2: Emo={final_emo}, Form={final_form}, Tempo={final_tempo}, DynShape={final_dynamic_shape}")

        midi_obj = self._build_multi_track_music(
            final_emo, analyzed_topic_val, final_form, final_melody_scale, final_tempo, 
            final_density, final_vr, final_mrh, final_prog, final_harmony_scale, final_is_major,
            final_dynamic_shape # Pass the new dynamic shape hint
        )
        
        midi_obj.save(out_path)
        print(f"▶️ Saved Multi-Track V2: {out_path} [{final_emo}/{analyzed_topic_val}/{final_form} @ {final_tempo} BPM]")

# --- Rest of Cell 1 (extract_feats, rule_emotion) remains the same ---
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
    if bpm>125 and dyn>65 and dens > 2.5: return "happy"
    if bpm<85 and dens < 2.5 and dyn < 60: return "sad"
    if dens<2.0 and bpm < 105 and dyn < 70 : return "calm"
    if bpm > 110 and dyn > 60 and dens > 2.0 : return "energetic" 
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
# Cell 3 (or your current cell number for this block): chapter → form+emotion → background MIDI
import re
from pathlib import Path

# --- MAKE SURE THIS LINE IS HERE AND RUNS BEFORE THE LOOP ---
gen = FormAwareTextToMusic() 
# -------------------------------------------------------------

OUT = Path("background_music_nb")
OUT.mkdir(exist_ok=True)

def split_into_chapters(txt): 
    segs = re.findall(
        r'(Chapter\s+\w+[\s\S]*?)(?=(?:Chapter\s+\w+)|\Z)',
        txt, flags=re.IGNORECASE
    )
    return segs if segs else [txt]

for book in Path("books").glob("*.txt"):
    text = book.read_text(encoding="utf-8")
    chaps = split_into_chapters(text)
    bd    = OUT / book.stem
    bd.mkdir(exist_ok=True)
    for i, ch in enumerate(chaps, start=1):
        # CORRECTED: Unpack all 12 values from analyze_text
        (analyzed_emo, analyzed_topic, analyzed_form, 
         analyzed_scale, analyzed_tempo, analyzed_density, 
         analyzed_vr, analyzed_mrh, 
         analyzed_prog_template, analyzed_harmony_roots, 
         analyzed_is_major, analyzed_dynamic_shape) = gen.analyze_text(ch) # Parentheses for readability
        
        outm_filename = bd / f"{book.stem}_chap{i}_{analyzed_form}_{analyzed_emo}.mid"
        
        gen.text_to_midi(ch, str(outm_filename), form_override=analyzed_form, emotion_override=analyzed_emo)
        print(f"→ {outm_filename.name}")

print("✅ All chapters generated under background_music_nb/")

# %%



