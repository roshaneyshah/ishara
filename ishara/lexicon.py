"""Roman Urdu tokens used to flag English-condition prompts that are not fully English.

The list is a flagger. It reports matches for a human to check. It does not decide.
Words that are also English words are removed (see ENGLISH_COLLISIONS) to avoid false alarms. Add any word you find in your own data.
"""
import re

ROMAN_URDU_TOKENS = {
    # verbs, auxiliaries, copula
    "hai", "hain", "hay", "hy", "tha", "thi", "thay", "thy", "hoga", "hogi", "hongay", "hota", "hoti",
    "hotay", "howa", "hua", "hui", "kar", "karo", "karna", "karni", "karta", "karti", "kartay",
    "karein", "karen", "kiya", "kia", "kiye", "raha", "rahi", "rahay", "rahe", "gaya", "gayi", "gae",
    "gaye", "jao", "jana", "jaana", "aao", "aana", "ana", "dena", "dedo", "lena", "lelo", "chalo",
    "chal", "dekho", "dekh", "suno", "sun", "bolo", "bol", "batao", "bata", "samjho", "samajh",
    "chahiye", "chahye", "sakta", "sakti", "saktay", "sakte", "wala", "wali", "walay", "wale",
    # pronouns and possessives
    "main", "mein", "mai", "mujhe", "mujhay", "mera", "meri", "meray", "mere", "tum", "tumhe",
    "tumhein", "tumhara", "tumhari", "aap", "ap", "apka", "apki", "aapka", "aapki", "aapko", "apko",
    "hum", "humein", "hamara", "hamari", "tera", "teri", "tujhe", "woh", "wo", "yeh", "ye", "uska",
    "uski", "unka", "unki", "usay", "use_ur", "inhe", "unhe", "kisi", "koi", "kuch", "sab",
    # postpositions and particles
    "ka", "ki", "ke", "ko", "se", "say", "sy", "bhi", "sirf", "hi_ur", "tou", "toh", "nahi",
    "nahin", "nai", "nhi", "mat", "haan", "jee", "ji", "acha", "achha", "accha", "theek", "thik",
    "yaar", "yar", "bas", "lekin", "magar", "agar", "phir", "fir", "kyun", "kyon", "kiun", "kya",
    "kia_q", "kaise", "kaisay", "kaisa", "kaisi", "kab", "kahan", "kidhar", "idhar", "udhar",
    "abhi", "kal", "aaj", "parson", "jaldi", "zara", "thora", "thoda", "thori", "bohat", "bahut",
    "bohot", "zyada", "ziyada", "kam", "wapas", "sath", "saath",
    # kinship and address terms
    "bhai", "bhaiya", "behen", "baji", "api", "ammi", "abbu", "ammi_jan", "khala", "phuppo",
    "chacha", "mamu", "nana", "nani", "dada", "dadi", "beta", "beti", "bachay", "bachon",
    # common nouns, adjectives and formulas
    "baat", "kaam", "ghar", "khana", "paisay", "paise", "shukriya", "meherbani", "maaf", "maafi",
    "inshallah", "insha", "mashallah", "alhamdulillah", "khuda", "hafiz", "salam", "assalam",
    "walaikum", "zaroor", "zarur", "bilkul", "sahi", "galat", "ghalat", "pareshan", "mushkil",
}

# Words that are also English words are removed so they never raise a flag.
ENGLISH_COLLISIONS = {"main", "say", "hay", "thy", "mat", "sun", "api", "ana", "ap", "hum", "fir",
                      "nana", "dada", "use", "hi", "to", "me", "the", "par", "na", "han", "chai"}
# Keys with an underscore were placeholders and never match real text.
ROMAN_URDU_TOKENS = {t for t in ROMAN_URDU_TOKENS if "_" not in t} - ENGLISH_COLLISIONS

_WORD = re.compile(r"[A-Za-z']+")


def roman_urdu_hits(text):
    """Return the Roman Urdu tokens found in text, in order of appearance, lower case."""
    hits = []
    for w in _WORD.findall(text or ""):
        lw = w.lower().strip("'")
        if lw in ROMAN_URDU_TOKENS:
            hits.append(lw)
    return hits
