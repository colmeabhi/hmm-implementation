---

# CS 271: Hidden Markov Models

This repository contains the source code and materials for **Assignment 1** of CS 271, a course on cryptography and network security taught by **Prof. Mark Stamp** at San José State University during the Fall 2025 semester. The assignment focuses on implementing and applying **Hidden Markov Models (HMMs)** to solve various problems, including probability computation and a cryptographic task.

---

## Project Structure

The project is organized into a clear directory structure for easy navigation.

├── src/
│   ├── Q_2_1.py            # HMM probability computation (Brute Force vs. Forward Algorithm)
│   ├── Q_2_3.py            # Normalization check for the Forward Algorithm
│   ├── Q_2_10.py           # HMM training using the Baum-Welch algorithm on English text
│   ├── Q_2_11.py           # Ciphertext decoding using an HMM
│   ├── Q_2_15.py           # Additional HMM analysis
│   ├── Q_10_2.py           # Letter2Vec embeddings using HMM emission matrix (N=3 and N=4)
│   ├── hmmMain.py          # Core HMM algorithms (Forward, Backward, Baum-Welch)
│   └── datasets.py         # Text preprocessing and encoding utilities
│
├── BrownCorpus.txt         # Text corpus used for HMM training
├── codeOutput.txt          # Sample output from key program runs
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Files to ignore in Git


---

## Key Concepts and Implementation

This project explores several core concepts related to HMMs:

* **Problem 2.1**: We implemented and compared two methods for computing the probability of an observation sequence, $P(O|\lambda)$:
    * **Brute-Force**: This naive approach calculates all possible state paths, which is computationally expensive. We verified the high operation count.
    * **Forward Algorithm**: A dynamic programming approach that drastically reduces the number of operations, demonstrating its efficiency for HMM evaluation.
* **Problem 2.3**: We validated the **normalization property** of the Forward Algorithm by summing the probabilities of all possible observation sequences, confirming that the total probability equals 1.0. This ensures the model is well-formed.
* **Problem 2.10**: We used the **Baum-Welch algorithm** to train HMMs on the **Brown Corpus**, a large text dataset. We trained models with varying numbers of states ($N$)—2, 3, 4, and 27—and reported the final log-likelihoods. This demonstrates how HMMs can learn the statistical properties of language.
* **Problem 2.11**: We applied HMMs to a **cryptanalysis** problem. The task involved decoding a ciphertext created with a substitution cipher. We used a two-step approach:
    1.  Trained an $N=2$ HMM to distinguish vowels from consonants.
    2.  Used this knowledge to initialize an $N=M=26$ HMM with a frozen transition matrix to infer the substitution key.
* **Problem 10.2**: We created **Letter2Vec** embeddings by training HMMs on English text with 27 observations (26 letters + space). The emission matrix $B$ provides vector representations for each letter:
    * **Problem 10.2a**: With $N=3$ hidden states, we generated 3-dimensional embeddings. The model separated vowels from consonants with average vowel similarity of 0.65.
    * **Problem 10.2b**: With $N=4$ hidden states, we generated 4-dimensional embeddings. This configuration successfully grouped all vowels together with much higher similarity (0.91), suggesting the HMM discovered a dedicated "vowel state."
    * We calculated cosine similarity matrices to group letters and analyzed the linguistic patterns captured by the embeddings.

---

## Getting Started

### Prerequisites

You need **Python 3.x** installed to run this project.

### Installation

First, clone the repository to your local machine:

```bash
git clone https://github.com/your-username/hmm-implementation.git
cd hmm-implementation
```

Next, install the required Python libraries using the requirements.txt file:

```bash
pip3 install -r requirements.txt
```

### Usage

Each problem is implemented in a separate Python script. Run the scripts directly to see the results:

```bash
# Calculate P(O|λ) using both methods and compare operation counts
python3 src/Q_2_1.py

# Verify the normalization of the Forward Algorithm
python3 src/Q_2_3.py

# Train HMMs on English text with different state counts
python3 src/Q_2_10.py

# Decode a ciphertext using the HMM approach
python3 src/Q_2_11.py

# Generate Letter2Vec embeddings (Problems 10.2a and 10.2b)
python3 src/Q_10_2.py
```
Sample Output
A sample of the program's output has been saved to codeOutput.txt for your convenience. This file demonstrates the expected results for each task.

Forward vs. Brute-Force:

P(O) by Bruteforce solving: 0.02488
P(O) by forward    : 0.02488
op counts, bf vs forward: 24 vs 18
Normalization Check:

sum P(O) by Bruteforce solving: 1.0
sum P(O) by forward    : 1.0
Checks passed.
Text Training:

N = 2 final logP = -285.99 chars = 109
N = 3 final logP = -280.58 chars = 109
N = 4 final logP = -265.48 chars = 109
N = 27 final logP = -142.0 chars = 109
Cipher Decoding:

[2.11b] vowel-like state index: 0
[2.11d] inferred key accuracy: 0.7692

Letter2Vec Embeddings (Problem 10.2):

**N=3 Hidden States:**
Letter groupings (threshold=0.65):
- Group 1: ['a', 'e', 'g', 'o', 'y']
- Group 2: ['b', 'c', 'd', 'f', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'z']
- Group 3: ['h', 'u', 'x', ' ']
- Group 4: ['i']

Average similarity within vowels: 0.6536
Average similarity within consonants: 0.8459

**N=4 Hidden States:**
Letter groupings (threshold=0.65):
- Group 1: ['a', 'e', 'i', 'o', 'u'] (all vowels grouped together!)
- Group 2: ['b', 'c', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'p', 'v', 'w', 'z']
- Group 3: ['d', 'n', 'r', 's', 'x', 'y', ' ']
- Group 4: ['q', 't']

Average similarity within vowels: 0.9110
Average similarity within consonants: 0.6114

---

## Letter2Vec: Detailed Analysis

The **Letter2Vec** approach (Problem 10.2) demonstrates how HMM emission matrices can be used to create meaningful letter embeddings:

### Methodology
1. **Data Preparation**: Load English text from the Brown Corpus and encode letters as observations (a=0, b=1, ..., z=25, space=26)
2. **HMM Training**: Train HMMs using the Baum-Welch algorithm with M=27 observations and N hidden states
3. **Embedding Extraction**: Transpose the emission matrix B to get letter embeddings (each row represents a letter's embedding)
4. **Similarity Analysis**: Calculate cosine similarity between all letter pairs to identify linguistic patterns

### Key Insights

**Impact of Hidden States:**
- **N=3**: Captures basic vowel/consonant distinction but with mixed groupings
- **N=4**: Produces linguistically meaningful groupings with all vowels clustering together (similarity 0.91)

**Linguistic Patterns Discovered:**
- Strong separation between vowels and consonants (cross-similarity < 0.22)
- Vowels form a tight cluster in the embedding space, especially with N=4
- Consonants show more diversity and form multiple sub-groups
- The HMM learns these patterns purely from statistical properties of English text

**Comparison to Word2Vec:**
- Similar to Word2Vec, Letter2Vec captures semantic relationships in an unsupervised manner
- The emission probabilities encode which letters commonly appear together in hidden states
- Higher dimensional embeddings (N=4) allow for more nuanced representations

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Author

Abhishek Ahirrao
CS 271 - Cryptography and Network Security
San José State University, Fall 2025
Instructor: Prof. Mark Stamp
