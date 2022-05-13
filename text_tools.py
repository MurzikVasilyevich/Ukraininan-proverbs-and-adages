import re


def text_cleaner(text):
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('  ', ' ')
    text = re.sub(r'^\s*[жз]\.*\s+', '', text)
    text = re.sub(r',\s*$', '.', text)
    text = re.sub(r"\s+[жз]\.*\s*$", '', text)
    text = re.sub(r" \w--  ", '', text)
    text = re.sub(r"^\W", '', text)
    text = re.sub(r"(\w)- (\w)", r'\1\2', text)
    text = re.sub(r"(\w)(--)(\s+)", r'\1 - ', text)
    text = re.sub(r"--", r'', text)
    text = " ".join(text.split())
    return text


def text_filter(text):
    digits = re.compile(r"^\d*$")
    non_words = re.compile(r"^\W*$")
    if digits.match(text):
        return False
    if len(text) < 3:
        return False
    if non_words.match(text):
        return False
    return True