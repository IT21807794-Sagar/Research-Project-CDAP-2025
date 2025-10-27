import re


# validation
def find_headings(text):
    headings = {}

    #  heading patterns
    patterns = [
        r'(?P<number>\d+)[\.\)\]]\s+(?P<subject>[^\n]+)(?=\n\d+[\.\)\]]|\Z)',
        r'(?:CHAPTER|Chapter|Part)\s+(?P<number>\d+)\s*[:\.]?\s*(?P<subject>[^\n]+)',
        r'(?P<subject>[A-Z][A-Z0-9\s]{5,})(?=\n)',
        r'(?P<subject>[A-Z][a-z]+(?: [A-Z][a-z]+)+)(?=\n)',
    ]

    for i, pattern in enumerate(patterns):
        matches = re.finditer(pattern, text)
        for match_num, match in enumerate(matches, 1):
            num = match.group('number') if 'number' in match.groupdict() else str(match_num)
            subject = match.group('subject').strip()
            headings[num] = subject

    return headings if headings else None