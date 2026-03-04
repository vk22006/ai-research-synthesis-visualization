def extract_claim(text):
    sentences = text.split(".")
    return sentences[0]