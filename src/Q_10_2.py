import numpy as np
import re
from hmmMain import initRandom, baumWelch

# Constants
ALPHABET = "abcdefghijklmnopqrstuvwxyz "  # 26 letters + space
M = 27  # Number of observations (26 letters + space)
IDX = {ch: i for i, ch in enumerate(ALPHABET)}

def cleanText(text):
    """Convert text to lowercase and keep only letters and spaces."""
    text = text.lower()
    text = re.sub(r'[^a-z ]+', ' ', text)  # Replace non-letters with spaces
    text = re.sub(r' +', ' ', text)  # Collapse multiple spaces
    return text

def encodeWithSpace(text):
    """Encode text to observation sequence where a=0, b=1, ..., z=25, space=26."""
    out = []
    for ch in text:
        if ch in IDX:
            out.append(IDX[ch])
    return np.array(out, dtype=int)

def loadEnglishText(path, max_chars=50000):
    """Load English text from file."""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read(max_chars)
    return cleanText(raw)

def cosineSimilarity(v1, v2):
    """Calculate cosine similarity between two vectors."""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def buildSimilarityMatrix(embeddings):
    """Build cosine similarity matrix for all letter embeddings."""
    n = len(embeddings)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = cosineSimilarity(embeddings[i], embeddings[j])
    return sim_matrix

def groupLetters(sim_matrix, threshold=0.65):
    """Group letters based on cosine similarity threshold."""
    n = sim_matrix.shape[0]
    visited = [False] * n
    groups = []

    for i in range(n):
        if visited[i]:
            continue

        group = [i]
        visited[i] = True

        for j in range(i + 1, n):
            if not visited[j] and sim_matrix[i, j] >= threshold:
                group.append(j)
                visited[j] = True

        groups.append(group)

    return groups

def printResults(N, B, threshold=0.65):
    """Print letter groupings and similarity statistics."""
    print(f"\n{'='*60}")
    print(f"Results for N={N} hidden states")
    print(f"{'='*60}")

    # Extract embeddings (transpose B so each row is a letter's embedding)
    embeddings = B.T  # Shape: (27, N)

    # Calculate similarity matrix
    sim_matrix = buildSimilarityMatrix(embeddings)

    # Group letters
    groups = groupLetters(sim_matrix, threshold)

    # Print groups
    print(f"\nLetter groupings (threshold={threshold}):")
    for idx, group in enumerate(groups, 1):
        letters = [ALPHABET[i] for i in group]
        print(f"Group {idx}: {letters}")

    # Calculate vowel group similarity
    vowels = [IDX[ch] for ch in 'aeiou']
    vowel_sims = []
    for i in vowels:
        for j in vowels:
            if i < j:
                vowel_sims.append(sim_matrix[i, j])

    if vowel_sims:
        avg_vowel_sim = np.mean(vowel_sims)
        print(f"\nAverage similarity within vowels (a,e,i,o,u): {avg_vowel_sim:.4f}")

    # Calculate consonant group similarity (common consonants)
    consonants = [IDX[ch] for ch in 'bcdfghjklmnpqrstvwxyz']
    consonant_sims = []
    for i in consonants:
        for j in consonants:
            if i < j:
                consonant_sims.append(sim_matrix[i, j])

    if consonant_sims:
        avg_consonant_sim = np.mean(consonant_sims)
        print(f"Average similarity within consonants: {avg_consonant_sim:.4f}")

    # Calculate cross-group similarity (vowels vs consonants)
    cross_sims = []
    for i in vowels:
        for j in consonants:
            cross_sims.append(sim_matrix[i, j])

    if cross_sims:
        avg_cross_sim = np.mean(cross_sims)
        print(f"Average similarity between vowels and consonants: {avg_cross_sim:.4f}")

    # Show some high similarity pairs
    print(f"\nTop 10 most similar letter pairs:")
    pairs = []
    for i in range(27):
        for j in range(i + 1, 27):
            pairs.append((i, j, sim_matrix[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    for i, j, sim in pairs[:10]:
        print(f"  {ALPHABET[i]} - {ALPHABET[j]}: {sim:.4f}")

def main():
    print("Problem 10.2: Letter2Vec using HMM Emission Matrix")
    print("="*60)

    # Load English text
    corpus_path = "/Users/abhishekahirrao/Documents/GitHub/hmm-implementation/BrownCorpus.txt"
    print(f"\nLoading text from {corpus_path}...")
    text = loadEnglishText(corpus_path, max_chars=20000)
    print(f"Loaded {len(text)} characters")

    # Encode text
    O = encodeWithSpace(text)
    print(f"Encoded to {len(O)} observations")

    # Problem 10.2a: N=3
    print(f"\n{'='*60}")
    print("Problem 10.2a: Training HMM with N=3 hidden states")
    print(f"{'='*60}")

    N = 3
    A, B, pi = initRandom(N, M, seed=42)
    print(f"Training HMM with N={N}, M={M}...")
    A, B, pi, logP = baumWelch(O, A, B, pi, iters=30, tol=1e-6)
    print(f"Training complete. Final log P(O|lambda) = {logP:.4f}")

    printResults(N, B, threshold=0.65)

    # Problem 10.2b: N=4
    print(f"\n\n{'='*60}")
    print("Problem 10.2b: Training HMM with N=4 hidden states")
    print(f"{'='*60}")

    N = 4
    A, B, pi = initRandom(N, M, seed=42)
    print(f"Training HMM with N={N}, M={M}...")
    A, B, pi, logP = baumWelch(O, A, B, pi, iters=30, tol=1e-6)
    print(f"Training complete. Final log P(O|lambda) = {logP:.4f}")

    printResults(N, B, threshold=0.65)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
