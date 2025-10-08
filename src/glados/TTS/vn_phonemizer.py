import re
from pathlib import Path
from pickle import load
from typing import Any


class Phonemizer:
    """
    Vietnamese Phonemizer using IPA (International Phonetic Alphabet).
    
    Based on Vietnamese phonology and compatible with VietnamesePhonotacticRulesAnalyzer.
    Converts Vietnamese text to IPA phonemes with proper tone marking.
    
    Vietnamese syllable structure: (Onset) (Glide) Nucleus (Coda) Tone
    """
    
    def __init__(self, phoneme_dict_path: Path | None = None):
        """
        Initialize the Vietnamese Phonemizer.
        
        Args:
            phoneme_dict_path: Optional path to a pickle file containing
                             a dictionary of word -> phoneme mappings
        """
        # Load external dictionary if provided
        self.phoneme_dict: dict[str, str] = {}
        if phoneme_dict_path and phoneme_dict_path.exists():
            with phoneme_dict_path.open("rb") as f:
                self.phoneme_dict = load(f)
        
        # Tone markers (6 tones in Vietnamese)
        self.tone_map = {
            "ngang": "1",  # Level tone (no mark)
            "huyền": "2",  # Grave/falling
            "ngã": "3",    # Tilde/rising glottalized
            "hỏi": "4",    # Hook/dipping-rising
            "sắc": "5",    # Acute/rising
            "nặng": "6",   # Dot below/falling glottalized
        }
        
        # Initial consonants (onset) and their IPA representations
        self.onset_map = {
            "b": "b",
            "m": "m",
            "n": "n",
            "ph": "f",
            "v": "v",
            "t": "t",
            "th": "tʰ",
            "đ": "d",
            "d": "z",
            "gi": "z",
            "r": "ʐ",
            "x": "s",
            "s": "ʂ",
            "ch": "c",
            "tr": "ʈ",
            "nh": "ɲ",
            "l": "l",
            "k": "k",
            "q": "k",
            "c": "k",
            "kh": "x",
            "ngh": "ŋ",
            "ng": "ŋ",
            "gh": "ɣ",
            "g": "ɣ",
            "h": "h",
            "": "?"  # Changed from "" to "?" for glottal stop (no onset)
        }
        
        # Map to remove tone marks from characters
        self.remove_tone_map = {
            'á': 'a', 'à': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
            'ắ': 'ă', 'ằ': 'ă', 'ẳ': 'ă', 'ẵ': 'ă', 'ặ': 'ă',
            'ấ': 'â', 'ầ': 'â', 'ẩ': 'â', 'ẫ': 'â', 'ậ': 'â',
            'é': 'e', 'è': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
            'ế': 'ê', 'ề': 'ê', 'ể': 'ê', 'ễ': 'ê', 'ệ': 'ê',
            'í': 'i', 'ì': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
            'ó': 'o', 'ò': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
            'ố': 'ô', 'ồ': 'ô', 'ổ': 'ô', 'ỗ': 'ô', 'ộ': 'ô',
            'ớ': 'ơ', 'ờ': 'ơ', 'ở': 'ơ', 'ỡ': 'ơ', 'ợ': 'ơ',
            'ú': 'u', 'ù': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
            'ứ': 'ư', 'ừ': 'ư', 'ử': 'ư', 'ữ': 'ư', 'ự': 'ư',
            'ý': 'y', 'ỳ': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
            'a': 'a', 'ă': 'ă', 'â': 'â',
            'e': 'e', 'ê': 'ê',
            'i': 'i',
            'o': 'o', 'ô': 'ô', 'ơ': 'ơ',
            'u': 'u', 'ư': 'ư',
            'y': 'y'
        }
        
        # Single-character onsets
        self.single_onset = {"b", "m", "v", "đ", "d", "r", "x", "s", "l", "q", "h", "t", "g", "c", "n", "k"}
        
        # Possible double and triple character onsets
        self.possible_double_onset = {"t", "g", "c", "n", "k"}
        self.double_onset = {"ph", "th", "ch", "tr", "nh", "kh", "ngh", "ng", "gh", "gi"}
        self.possible_triple_onset = {"ng"}
        self.triple_onset = {"ngh"}
        
        # Glide map (semivowels) - using combining diacritic for non-syllabic
        self.glide_map = {
            "o": "w̆",
            "u": "w̆",
        }
        
        # Nucleus (main vowel) map
        self.nucleus_map = {
            "y": "i",
            "i": "i",
            "ê": "e",
            "e": "ɛ",
            "ư": "ɯ",
            "u": "u",
            "ơ": "ɤ",
            "ô": "o",
            "ôô": "o",
            "o": "ɔ̆",
            "oo": "ɔ",
            "a": {
                "u": "ă",
                "y": "ă",
                "nh": "ɛ̆",
                "ch": "ɛ̆"
                # default "a"
            },
            "ă": "ă",
            "iê": "iə",
            "yê": "iə",
            "ia": "iə",
            "ya": "iə",
            "ươ": "ɯɤ",
            "ưa": "ɯɤ",
            "uô": "uə",
            "ua": "uə",
            "â": "ɤ̃"
        }
        
        # Glide-nucleus mapping (determines when a character is a glide vs nucleus)
        self.glide_nucleus_mapping = {
            "o": {
                "e", "é", "è", "ẽ", "ẻ", "ẹ",
                "a", "á", "à", "ã", "ả", "ạ",
                "ă", "ắ", "ằ", "ẵ", "ẳ", "ặ"
            },
            "u": {
                "y", "ý", "ỳ", "ỹ", "ỷ", "ỵ",
                "ê", "ế", "ề", "ễ", "ể", "ệ",
                "ơ", "ớ", "ờ", "ỡ", "ở", "ợ",
                "â", "ấ", "ầ", "ẫ", "ẩ", "ậ"
            }
        }
        
        # Final consonants (coda)
        self.coda_map = {
            "m": "m",
            "n": "n",
            "p": "p",
            "t": "t",
            "nh": "ŋ",
            "ng": "ŋ",
            "ch": "k",
            "c": "k",
            "o": "w",
            "u": "w",
            "y": "j",
            "i": "j"
        }
    
    def check_tone(self, char: str) -> str | None:
        """
        Detect tone from a Vietnamese character with tone mark.
        
        Args:
            char: Single Vietnamese character
            
        Returns:
            Tone name or None if no tone mark
        """
        if char in "áắấéếóốíúớứý":
            return "sắc"
        elif char in "àằầòồèềìùờừỳ":
            return "huyền"
        elif char in "ãẫẵõỗĩẽễũỡữỹ":
            return "ngã"
        elif char in "ảẩẳẻểỏổỉủởửỷ":
            return "hỏi"
        elif char in "ạặậịọộẹệụợựỵ":
            return "nặng"
        return None
    
    def find_onset(self, word: str) -> str:
        """
        Find the initial consonant (onset) of a Vietnamese syllable.
        
        Args:
            word: Vietnamese syllable
            
        Returns:
            Onset string (empty if no onset)
        """
        word = word.lower()
        
        # Check for 3-character onset
        if len(word) >= 3 and word[:3] in self.triple_onset:
            return word[:3]
        
        # Check for 2-character onset (including 'gi')
        if len(word) >= 2 and word[:2] in self.double_onset:
            return word[:2]
        
        # Check for 1-character onset
        if word and word[0] in self.single_onset:
            return word[0]
        
        return ""
    
    def find_glide(self, word: str, onset: str) -> str:
        """
        Find the glide (semivowel) in a Vietnamese syllable.
        
        Args:
            word: Remaining part of syllable after onset
            onset: The onset that was found
            
        Returns:
            Glide string (empty if no glide)
        """
        if not word:
            return ""
        
        val = word[:1]
        if val not in self.glide_map:
            return ""
        
        # Special handling for 'q' onset (always has 'u' glide)
        if onset == "q":
            if val == "u":
                return val
            else:
                raise ValueError(f"Invalid word: 'q' must be followed by 'u'")
        
        # Check if next character indicates this is a glide
        if len(word) < 2:
            return ""
        
        next_char = word[1]
        if next_char in self.glide_nucleus_mapping.get(val, set()):
            return val
        
        return ""
    
    def find_nucleus(self, word: str) -> str:
        """
        Find the nucleus (main vowel) of a Vietnamese syllable.
        
        Args:
            word: Remaining part of syllable after onset and glide
            
        Returns:
            Nucleus string
        """
        if len(word) >= 2:
            val = word[:2]
            if val.lower() in self.nucleus_map:
                return val
        
        return word[:1] if word else ""
    
    def _phonemize_syllable(self, word: str) -> str:
        """
        Convert a single Vietnamese syllable to IPA phonemes.
        
        Args:
            word: Vietnamese syllable
            
        Returns:
            IPA phoneme representation with tone marker
        """
        # Check dictionary first
        word_lower = word.lower()
        if word_lower in self.phoneme_dict:
            return self.phoneme_dict[word_lower]
        
        # Handle punctuation-only or empty
        if not word or not any(c.isalpha() for c in word):
            return word
        
        # Detect tone
        tone = "ngang"
        for c in word:
            tone_detected = self.check_tone(c)
            if tone_detected is not None:
                tone = tone_detected
        
        # Remove tone marks
        word_list = [self.remove_tone_map.get(c, c) for c in word]
        word_normalized = "".join(word_list)
        
        # Parse syllable structure
        onset = self.find_onset(word_normalized)
        word_normalized = word_normalized.removeprefix(onset)
        
        glide = self.find_glide(word_normalized, onset)
        word_normalized = word_normalized.removeprefix(glide)
        
        nucleus = self.find_nucleus(word_normalized)
        coda = word_normalized.removeprefix(nucleus)
        
        # Handle trailing punctuation
        extra = ""
        if coda and coda[-1] in {",", ".", "!", "?", ";", ":", "-"}:
            extra = coda[-1]
            coda = coda[:-1]
        
        # Build IPA representation
        result = ""
        
        # Add onset (use '?' for glottal stop if no onset)
        if onset:
            result += self.onset_map.get(onset.lower(), onset)
        else:
            # Only add '?' if there's a nucleus (valid syllable)
            if nucleus:
                result += "?"
        
        # Add glide
        result += self.glide_map.get(glide.lower(), "")
        
        # Handle nucleus (special case for 'a' which changes based on coda)
        nucleus_value = self.nucleus_map.get(nucleus.lower(), nucleus.lower())
        if isinstance(nucleus_value, dict):
            # Look up coda-specific variant, default to 'a'
            nucleus_val = nucleus_value.get(coda.lower(), "a")
            result += nucleus_val
        else:
            result += nucleus_value
        
        # Add coda
        result += self.coda_map.get(coda.lower(), "")
        
        # Add tone marker
        result += self.tone_map[tone.lower()]
        
        # Add back punctuation
        result += extra
        
        return result
    
    def phonemize(self, text: str) -> str:
        """
        Convert Vietnamese text to IPA phonemes.
        
        Args:
            text: Vietnamese text
            
        Returns:
            IPA phoneme representation with spaces between syllables
        """
        text = text.strip()
        
        # Split into syllables (Vietnamese words are separated by spaces)
        syllables = []
        current = ""
        for char in text:
            if char == " ":
                if current:
                    syllables.append(current)
                    current = ""
            else:
                current += char
        
        if current:
            syllables.append(current)
        
        # Phonemize each syllable
        result = []
        for syllable in syllables:
            try:
                phonemized = self._phonemize_syllable(syllable)
                result.append(phonemized)
            except Exception as e:
                # If phonemization fails, keep original syllable
                print(f"Warning: Failed to phonemize '{syllable}': {e}")
                result.append(syllable)
        
        return " ".join(result)
    
    def convert_to_phonemes(self, texts: list[str], lang: str = "vi") -> list[str]:
        """
        Convert a list of Vietnamese texts to phonemes.
        
        Args:
            texts: List of input texts
            lang: Language code (default: "vi" for Vietnamese)
            
        Returns:
            List of phoneme strings
        """
        return [self.phonemize(text) for text in texts]
    
    def __call__(self, texts: list[str]) -> list[str]:
        """
        Convenience method to call convert_to_phonemes.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of phoneme strings
        """
        return self.convert_to_phonemes(texts)
    
    def count_possible_syllables(self) -> int:
        """
        Compute number of possible Vietnamese syllables (by formula).
        
        Returns:
            Theoretical maximum number of syllables
        """
        n_onset = len(self.onset_map.keys())
        n_glide = len(self.glide_map.keys()) + 1   # +1 for no-glide
        n_nucleus = len(self.nucleus_map.keys())
        n_coda = len(self.coda_map.keys()) + 1     # +1 for no-coda
        n_tone = len(self.tone_map.keys())
        
        return n_onset * n_glide * n_nucleus * n_coda * n_tone


# Example usage and testing
if __name__ == "__main__":
    phonemizer = VietnamesePhonemizer()
    
    # Test with some Vietnamese text
    test_cases = [
        "Xin chào",
        "Tôi là trợ lý ảo",
        "Hôm nay thời tiết đẹp",
        "Chúc bạn một ngày tốt lành",
        "Việt Nam",
        "Hà Nội",
        "quả",  # Test 'qu' onset
        "ăn",   # Test glottal stop (no onset)
    ]
    
    print("Vietnamese Phonemizer Test:")
    print("=" * 70)
    for text in test_cases:
        phonemes = phonemizer.phonemize(text)
        print(f"{text:30} -> {phonemes}")
    
    print(f"\n{'=' * 70}")
    print(f"Total possible syllables: {phonemizer.count_possible_syllables():,}")
    print(f"Number of onsets: {len(phonemizer.onset_map)}")
    print(f"Number of glides: {len(phonemizer.glide_map) + 1} (+1 for no-glide)")
    print(f"Number of nuclei: {len(phonemizer.nucleus_map)}")
    print(f"Number of codas: {len(phonemizer.coda_map) + 1} (+1 for no-coda)")
    print(f"Number of tones: {len(phonemizer.tone_map)}")