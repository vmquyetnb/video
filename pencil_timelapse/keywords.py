"""Offline keyword suggestions for finding scene pictures.

Image archives are searched in English. Vietnamese narration is matched
against a phrase dictionary (education, health, family, daily life,
environment); English narration is reduced to its visual words (people,
actions, objects, places). The suggestion is only a starting point; it can be
edited before rendering.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Tuple

# Longer phrases win over the words they contain ("rửa tay" before "tay").
# An empty translation swallows a phrase that would otherwise mislead
# ("nghỉ học" = to miss school, not "studying").
PHRASES = {
    "nghỉ học": "", "học lớp": "", "cả lớp": "", "hôm nay": "", "ngày mai": "",
    # people
    "học sinh": "students", "sinh viên": "university students", "cô giáo": "teacher",
    "thầy giáo": "teacher", "giáo viên": "teacher", "thầy cô": "teachers",
    "bác sĩ": "doctor", "y tá": "nurse", "nha sĩ": "dentist", "dược sĩ": "pharmacist",
    "bệnh nhân": "patient", "em bé": "baby", "trẻ sơ sinh": "newborn baby", "trẻ em": "children",
    "trẻ con": "children", "các em": "children", "con cái": "children", "cậu bé": "boy",
    "bé trai": "boy", "cô bé": "girl", "bé gái": "girl", "thiếu niên": "teenagers",
    "thanh niên": "young people", "người già": "elderly", "cụ già": "elderly person",
    "người cao tuổi": "elderly", "bà cụ": "elderly woman", "cụ bà": "elderly woman",
    "ông cụ": "elderly man", "cụ ông": "elderly man", "người phụ nữ": "woman",
    "người đàn ông": "man", "ông bà": "grandparents", "ông nội": "grandfather",
    "ông ngoại": "grandfather", "bà nội": "grandmother", "bà ngoại": "grandmother",
    "bố mẹ": "parents", "cha mẹ": "parents", "phụ huynh": "parents", "người mẹ": "mother",
    "mẹ": "mother", "người cha": "father", "bố": "father", "anh chị em": "siblings",
    "gia đình": "family", "bạn bè": "friends", "bạn thân": "friends", "hàng xóm": "neighbors",
    "cộng đồng": "community", "tình nguyện viên": "volunteers", "nông dân": "farmer",
    "công nhân": "worker", "cảnh sát": "police officer", "người lớn": "adults",
    # school and learning
    "lớp học": "classroom", "trường học": "school", "sân trường": "schoolyard",
    "giờ ra chơi": "school recess", "thư viện": "library", "đọc sách": "reading book",
    "quyển sách": "book", "bài tập về nhà": "homework", "bài tập": "homework", "bài học": "lesson",
    "kỳ thi": "exam", "thi cử": "exam", "tốt nghiệp": "graduation", "học tập": "studying",
    "học bài": "studying", "giáo dục": "education", "bảng đen": "blackboard", "viết bài": "writing",
    "máy tính": "computer", "điện thoại thông minh": "smartphone", "điện thoại": "mobile phone",
    "mạng xã hội": "social media", "trò chơi điện tử": "video games", "internet": "internet",
    "đi học": "going to school", "trường": "school", "học": "studying",
    # health and body
    "bệnh viện": "hospital", "phòng khám": "clinic", "trạm y tế": "health center",
    "khám bệnh": "medical examination", "tiêm phòng": "vaccination", "tiêm chủng": "vaccination",
    "vắc xin": "vaccine", "vaccine": "vaccine", "tiêm": "injection", "uống thuốc": "taking medicine",
    "thuốc lá": "smoking", "hút thuốc": "smoking", "thuốc": "medicine", "đơn thuốc": "prescription",
    "sốt": "fever", "ho": "coughing", "cảm cúm": "flu", "cúm": "flu", "đau bụng": "stomach ache",
    "đau đầu": "headache", "đau răng": "toothache", "khẩu trang": "face mask",
    "rửa tay": "washing hands", "xà phòng": "soap", "đánh răng": "brushing teeth",
    "vệ sinh": "hygiene", "tắm": "bathing", "sức khỏe": "health", "sức khoẻ": "health",
    "khỏe mạnh": "healthy", "khoẻ mạnh": "healthy", "bị ốm": "sick", "ốm": "sick",
    "bệnh": "illness", "mệt mỏi": "tired", "căng thẳng": "stress", "lo lắng": "worried",
    "trầm cảm": "depression", "đo huyết áp": "measuring blood pressure", "huyết áp": "blood pressure", "tim mạch": "heart health",
    "béo phì": "obesity", "cân nặng": "weight scale", "mang thai": "pregnant woman",
    "giấc ngủ": "sleep", "ngủ": "sleeping", "thức khuya": "staying up late",
    "tập thể dục": "exercise", "thể dục": "exercise", "chạy bộ": "jogging", "đi bộ": "walking",
    "bơi": "swimming", "yoga": "yoga", "thể thao": "sports", "bóng đá": "football",
    # food
    "dinh dưỡng": "healthy food", "rau xanh": "vegetables", "rau": "vegetables",
    "trái cây": "fruit", "hoa quả": "fruit", "đồ ăn nhanh": "fast food", "đồ ngọt": "sweets",
    "nước ngọt": "soft drinks", "uống nước": "drinking water", "nước sạch": "clean water",
    "sữa": "milk", "bữa sáng": "breakfast", "bữa cơm": "family meal", "bữa ăn": "meal",
    "ăn cơm": "eating meal", "ăn uống": "eating", "nấu ăn": "cooking", "nhà bếp": "kitchen",
    "bếp": "kitchen", "chợ": "market", "cơm": "rice meal", "ăn": "eating",
    # feelings and actions
    "vui vẻ": "happy", "hạnh phúc": "happy", "vui": "happy", "cười": "smiling", "buồn": "sad",
    "khóc": "crying", "tức giận": "angry", "sợ hãi": "scared", "cô đơn": "lonely",
    "giúp đỡ": "helping", "chia sẻ": "sharing", "ôm": "hugging", "nói chuyện": "talking",
    "lắng nghe": "listening", "trò chuyện": "conversation", "suy nghĩ": "thinking",
    "làm việc": "working", "chơi đùa": "playing", "vui chơi": "playing", "chơi": "playing",
    "hát": "singing", "âm nhạc": "music", "nhảy múa": "dancing", "trồng cây": "planting trees",
    "dọn dẹp": "cleaning", "quét": "sweeping", "giặt": "laundry",
    # more health, school and life situations
    "cấp cứu": "emergency room", "xe cứu thương": "ambulance", "tai nạn": "accident",
    "phẫu thuật": "surgery", "giường bệnh": "hospital bed", "xét nghiệm": "medical test",
    "tiểu đường": "blood sugar test", "đường huyết": "blood sugar test", "ung thư": "cancer patient",
    "dịch bệnh": "epidemic", "virus": "virus", "vi khuẩn": "bacteria", "cách ly": "quarantine",
    "cận thị": "child wearing glasses", "đeo kính": "wearing glasses", "mắt": "eyes",
    "răng miệng": "dental care", "sâu răng": "tooth decay", "tai": "ear", "tim": "heart",
    "sinh con": "newborn baby", "cho con bú": "breastfeeding", "sữa mẹ": "breastfeeding",
    "đuối nước": "swimming lesson", "học bơi": "swimming lesson", "sơ cứu": "first aid",
    "bắt nạt": "bullying", "bạo lực học đường": "school bullying", "áp lực": "stressed student",
    "điểm kém": "sad student", "điểm cao": "happy student", "học trực tuyến": "online learning",
    "học online": "online learning", "nghiện game": "video games", "chơi game": "video games",
    "kể chuyện": "storytelling", "đọc truyện": "reading", "vẽ tranh": "child painting",
    "học sinh tiểu học": "primary school children", "mẫu giáo": "kindergarten",
    "nhà trẻ": "kindergarten", "đại học": "university", "học bổng": "scholarship",
    "nghèo": "poverty", "khó khăn": "hardship", "lao động": "labor", "đồng ruộng": "farm field",
    "bữa trưa": "lunch", "bữa tối": "dinner", "rửa rau": "washing vegetables",
    "đun sôi": "boiling water", "nước máy": "tap water", "nhà vệ sinh": "toilet",
    "ngủ gật": "sleepy student", "nhịn ăn": "skipping breakfast", "ngủ trưa": "nap", "buổi sáng": "", "buổi tối": "", "buổi trưa": "", "mỗi sáng": "",
    # places and environment
    "ngôi nhà": "house", "nhà": "home", "làng quê": "rural village", "nông thôn": "countryside",
    "làng": "village", "thành phố": "city", "đường phố": "street", "công viên": "park",
    "cánh đồng": "rice field", "ruộng lúa": "rice paddy", "ruộng": "rice field", "núi": "mountains",
    "biển": "beach", "sông": "river", "hồ": "lake", "rừng": "forest", "cây xanh": "trees",
    "cây": "tree", "hoa": "flowers", "mưa": "rain", "nắng": "sunshine", "môi trường": "environment",
    "rác thải": "garbage", "rác": "garbage", "ô nhiễm": "pollution", "nhựa": "plastic waste",
    "giao thông": "traffic", "an toàn giao thông": "road safety", "mũ bảo hiểm": "helmet",
    "xe đạp": "bicycle", "xe máy": "motorbike", "xe buýt": "bus", "ô tô": "car",
}
_ORDERED = sorted(PHRASES.items(), key=lambda kv: -len(kv[0]))
# Generic actions and moods describe a picture less well than people, places
# and objects; they are only used when little else was found.
_GENERIC = {"studying", "eating", "playing", "happy", "healthy", "health", "home", "sad",
            "smiling", "talking", "thinking", "working", "helping", "sharing", "listening",
            "illness", "sick", "tired", "school", "going to school", "education", "morning",
            "evening", "sunrise", "hardship"}
# Places set the scene but a picture is chosen by who is in it and what they do,
# so places go after people and actions in the query.
_PLACES = {"hospital", "clinic", "health center", "school", "classroom", "schoolyard", "library",
           "home", "house", "kitchen", "market", "park", "village", "rural village",
           "countryside", "city", "street", "rice field", "rice paddy", "mountains", "beach",
           "river", "lake", "forest"}


# English stories: visual vocabulary with a role for each term so a query reads
# like a photo caption - who (P), doing what (A), with what / where (O, L).
EN_TERMS = {
    # people
    "boy": ("boy", "P"), "girl": ("girl", "P"), "kid": ("children", "P"), "child": ("child", "P"),
    "children": ("children", "P"), "son": ("boy", "P"), "daughter": ("girl", "P"),
    "man": ("man", "P"), "woman": ("woman", "P"), "old man": ("old man", "P"),
    "old woman": ("old woman", "P"), "elderly": ("elderly person", "P"),
    "grandfather": ("grandfather", "P"), "grandmother": ("grandmother", "P"),
    "mother": ("mother", "P"), "mom": ("mother", "P"), "mum": ("mother", "P"),
    "father": ("father", "P"), "dad": ("father", "P"), "parent": ("parents", "P"),
    "family": ("family", "P"), "families": ("families", "P"), "teacher": ("teacher", "P"),
    "student": ("student", "P"), "pupil": ("student", "P"), "classmate": ("classmates", "P"),
    "friend": ("friends", "P"), "doctor": ("doctor", "P"), "nurse": ("nurse", "P"),
    "patient": ("patient", "P"), "engineer": ("engineer", "P"), "mechanic": ("mechanic", "P"),
    "farmer": ("farmer", "P"), "worker": ("worker", "P"), "mayor": ("mayor", "P"),
    "police": ("police officer", "P"), "firefighter": ("firefighter", "P"),
    "neighbor": ("neighbors", "P"), "neighbour": ("neighbors", "P"), "people": ("people", "P"),
    "crowd": ("crowd", "P"), "baby": ("baby", "P"), "teenager": ("teenager", "P"),
    "young man": ("young man", "P"), "volunteer": ("volunteers", "P"),
    # actions and feelings
    "laughed": ("laughing", "A"), "laughing": ("laughing", "A"), "laugh": ("laughing", "A"),
    "cried": ("crying", "A"), "crying": ("crying", "A"), "tears": ("crying", "A"),
    "smiled": ("smiling", "A"), "smiling": ("smiling", "A"), "sad": ("sad", "A"),
    "sadly": ("sad", "A"), "exhausted": ("tired", "A"), "tired": ("tired", "A"),
    "angry": ("angry", "A"), "proud": ("proud", "A"), "hug": ("hugging", "A"),
    "hugged": ("hugging", "A"), "clean": ("cleaning", "A"), "cleaning": ("cleaning", "A"),
    "read": ("reading", "A"), "reading": ("reading", "A"), "writing": ("writing", "A"),
    "studying": ("studying", "A"), "studied": ("studying", "A"), "learning": ("learning", "A"),
    "teaching": ("teaching", "A"), "taught": ("teaching", "A"), "fix": ("repairing", "A"),
    "fixed": ("repairing", "A"), "repair": ("repairing", "A"), "repairing": ("repairing", "A"),
    "adjusted": ("repairing", "A"), "built": ("building", "A"), "building machines": ("machines", "O"),
    "running": ("running", "A"),
    "rushed": ("running", "A"), "escaped": ("evacuation", "A"), "sitting": ("sitting", "A"),
    "celebrated": ("celebration", "A"), "celebration": ("celebration", "A"),
    "applause": ("applause", "A"), "thinking": ("thinking", "A"), "thought": ("thinking", "A"),
    "praying": ("praying", "A"), "playing": ("playing", "A"), "cooking": ("cooking", "A"),
    "eating": ("eating", "A"), "sleeping": ("sleeping", "A"), "walking": ("walking", "A"),
    "helping": ("helping", "A"), "helped": ("helping", "A"), "lowered his head": ("sad", "A"),
    # objects
    "test paper": ("test paper", "O"), "test": ("exam", "O"), "exam": ("exam", "O"),
    "grade": ("test paper", "O"), "homework": ("homework", "O"), "notebook": ("notebook", "O"),
    "book": ("books", "O"), "backpack": ("backpack", "O"), "desk": ("desk", "O"),
    "blackboard": ("blackboard", "O"), "mathematics": ("mathematics", "O"), "math": ("mathematics", "O"),
    "clock": ("antique clock", "O"), "wristwatch": ("watch", "O"), "gear": ("gears", "O"),
    "wheel": ("gears", "O"), "spring": ("clockwork", "O"), "tool": ("tools", "O"),
    "machine": ("machines", "O"), "motor": ("electric motor", "O"), "battery": ("battery", "O"),
    "car battery": ("car battery", "O"), "wire": ("electrical wires", "O"),
    "electricity": ("electricity", "O"), "bicycle": ("bicycle", "O"), "bike": ("bicycle", "O"),
    "engine": ("engine", "O"), "siren": ("siren", "O"), "alarm": ("alarm siren", "O"),
    "computer": ("computer", "O"), "phone": ("mobile phone", "O"), "food": ("food", "O"),
    "medicine": ("medicine", "O"),
    "storm": ("storm", "O"), "rain": ("rain", "O"), "rainy": ("rain", "O"), "flood": ("flood", "O"),
    "flooding": ("flood", "O"), "overflowed": ("flood", "O"), "rising water": ("flood", "O"),
    "snow": ("snow", "O"), "fire": ("fire", "O"), "night": ("night", "O"),
    # places
    "classroom": ("classroom", "L"), "class": ("classroom", "L"), "school": ("school", "L"),
    "library": ("library", "L"), "home": ("home", "L"), "house": ("house", "L"),
    "kitchen": ("kitchen", "L"), "repair shop": ("repair shop", "L"), "workshop": ("workshop", "L"),
    "shop": ("shop", "L"), "hospital": ("hospital", "L"), "town": ("small town", "L"),
    "village": ("village", "L"), "city": ("city", "L"), "street": ("street", "L"),
    "river": ("river", "L"), "farm": ("farm", "L"), "park": ("park", "L"),
    "higher ground": ("evacuation", "A"),
}
_EN_ORDERED = sorted(EN_TERMS.items(), key=lambda kv: -len(kv[0]))
# Vietnamese letters that English never uses.
_VI_LETTERS = re.compile(r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụ"
                         r"ừứửữựỳýỷỹỵ]", re.I)


def is_vietnamese(text: str) -> bool:
    letters = len(re.findall(r"[^\W\d_]", text))
    return len(_VI_LETTERS.findall(text)) >= max(2, 0.02 * letters)


def english_terms(text: str) -> List[Tuple[str, str]]:
    """(term, role) for the visual words in an English text, in order of appearance."""
    lower = text.lower().replace("’", "'")
    taken = [False] * len(lower)
    found: List[Tuple[int, str, str]] = []
    for phrase, (term, role) in _EN_ORDERED:
        suffix = "(?:s|es)?" if " " not in phrase and role in "POL" else ""
        for m in re.finditer(r"(?<![\w'])" + re.escape(phrase) + suffix + r"(?![\w'])", lower):
            if any(taken[m.start():m.end()]):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            found.append((m.start(), term, role))
    out: List[Tuple[str, str]] = []
    for _, term, role in sorted(found):
        if all(term != t for t, _ in out):
            out.append((term, role))
    return out


def _english_query(text: str) -> str:
    """A caption-like query: who, doing what, with what or where."""
    terms = english_terms(text)
    if not terms:
        return ""
    first = {}
    for term, role in terms:
        first.setdefault(role, term)
    main = [first[r] for r in ("P", "A", "O") if r in first]
    for term, role in terms:  # fill up with the next people/objects/places
        if len(main) >= 3:
            break
        if term not in main and role in "POL":
            main.append(term)
    options = [" ".join(main[:3]), " ".join(main[:2]), main[0]]
    return " | ".join(dict.fromkeys(o for o in options if o))


def keyword_terms(text: str) -> List[str]:
    """English terms for the phrases found in ``text``, in order of appearance."""
    if not is_vietnamese(text):
        return [term for term, _ in english_terms(text)]
    lower = text.lower()
    taken = [False] * len(lower)
    found: List[Tuple[int, str]] = []
    for phrase, english in _ORDERED:
        for m in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", lower):
            if any(taken[m.start():m.end()]):
                continue
            for i in range(m.start(), m.end()):
                taken[i] = True
            if english:
                found.append((m.start(), english))
    terms: List[str] = []
    for _, english in sorted(found):
        if english not in terms:
            terms.append(english)
    return terms


def story_query(texts: Iterable[str]) -> str:
    """The story's main subject (most frequent people/objects), used for scenes
    whose own narration names nothing that can be photographed."""
    counts: Counter = Counter()
    first = {}
    for i, text in enumerate(texts):
        for term in keyword_terms(text):
            counts[term] += 1
            first.setdefault(term, i)
    ranked = sorted(counts, key=lambda t: (t in _GENERIC, t in _PLACES, -counts[t], first[t]))
    if not ranked:
        return ""
    return " | ".join(dict.fromkeys([" ".join(ranked[:2]), ranked[0]]))


_NAME = re.compile(r"\b(?:(?:Mr|Mrs|Ms|Dr)\.\s)?[A-Z][a-z]{2,}\b")
# Capitalised only because they start a sentence, never names.
_NOT_NAMES = {"then", "but", "and", "every", "one", "when", "now", "years", "sometimes", "instead",
              "because", "maybe", "without", "like", "if", "after", "before", "while", "once",
              "today", "tomorrow", "yesterday", "there", "this", "that", "these", "those", "what",
              "how", "why", "who", "where", "for", "from", "with", "yes", "no", "dear", "god"}
# Sentence ends, but not after a title such as "Mr." in "Mr. Thomas".
_SENTENCE_END = re.compile(r"(?<!\bMr\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bMrs\.)(?<=[.!?])\s+|\n+")


def character_roles(text: str) -> dict:
    """Recurring names in an English story mapped to who they are.

    A name is a capitalised word that never appears in lower case and occurs
    at least twice; its role is the first person word in a sentence that
    mentions it ("Daniel was ... the worst student" -> student, "The old man
    was Mr. Thomas" -> old man).
    """
    lower_words = set(re.findall(r"\b[a-z]+\b", text))
    counts = Counter(m.group(0).split()[-1] for m in _NAME.finditer(text))
    names = [n for n, c in counts.items() if c >= 2 and n.lower() not in lower_words
             and n.lower() not in EN_TERMS and n.lower() not in _NOT_NAMES]
    roles = {}
    for sentence in _SENTENCE_END.split(text):
        for name in names:
            if name in roles or not re.search(rf"\b{name}\b", sentence):
                continue
            people = [term for term, role in english_terms(sentence) if role == "P"]
            if people:
                roles[name] = people[0]
    return roles


def suggest_queries(scenes: List[str], title: str = "") -> List[str]:
    """Queries for a whole story, scene by scene.

    English scenes without a person in them get the character they talk about
    (or the last one mentioned before). Scenes that name nothing to photograph
    continue the previous picture's subject, or else the story's main subject.
    """
    full = "\n".join([title] + scenes)
    english = not is_vietnamese(full)
    roles = character_roles(full) if english else {}
    fallback = story_query([title] + scenes)
    queries: List[str] = []
    last_name = None
    for scene in scenes:
        mentioned = sorted((m.start(), n) for n in roles for m in [re.search(rf"\b{n}\b", scene)] if m)
        own = suggest_query(scene)
        if english:
            terms = english_terms(scene)
            who = roles.get(mentioned[0][1]) if mentioned else roles.get(last_name)
            if who and not any(role == "P" for _, role in terms):
                own = _english_query(f"{who} " + scene) if terms else who
        if mentioned:
            last_name = mentioned[-1][1]
        previous = queries[-1].split(" | ")[0] if queries and queries[-1] else ""
        queries.append(own or previous or fallback)
    return queries


def suggest_query(text: str, max_terms: int = 3) -> str:
    """A search query (with a shorter alternative after ``|``), or "" if nothing matched.

    Vietnamese text is translated with the phrase dictionary; English text is
    reduced to its visual words.
    """
    if not is_vietnamese(text):
        return _english_query(text)
    found = keyword_terms(text)
    specific = [t for t in found if t not in _GENERIC and t not in _PLACES]
    places = [t for t in found if t in _PLACES]
    generic = [t for t in found if t in _GENERIC and t not in _PLACES]
    ordered = specific + places + generic
    # "measuring blood pressure" already says "blood pressure"
    ordered = [t for t in ordered if not any(t != o and t in o for o in ordered)]
    terms = ordered[:max_terms]
    if not terms:
        return ""
    options: List[str] = []
    for option in (" ".join(terms), " ".join(terms[:2]), terms[0]):
        if option not in options:
            options.append(option)
    return " | ".join(options)
