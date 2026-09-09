---
title: CS Degree Prep - Semester 1 Checklist
date: 2026-09-04
tags: [cs, degree, algorithms, discrete-math]
---

# B.Sc. Computer Science — prep notes

## Admission / admin
- [ ] Verify IGNOU / state-university registration window closes 31 Oct.
- [ ] Upload 10th + 12th marksheets, migration certificate, caste certificate.
- [ ] Pay the ₹3,500 examination form fee before the 25th (late fee is 100 %).
- [ ] Request the practical-record format from the lab in-charge.

## Math foundations (needed before Data Structures)
1. **Discrete math** — sets, relations, functions, inclusion–exclusion.
   - `|A ∪ B| = |A| + |B| - |A ∩ B|`
   - Equivalence relation = reflexive + symmetric + transitive.
2. **Modular arithmetic** — Euclid's algorithm for `gcd`, inverse mod p.
3. **Combinatorics** — `nCr = n!/(r!(n-r)!)`, pigeonhole principle.
4. **Induction** — every recurrence proof in the exam is a two-line induction.

## Programming plan (C → Python → Java order matters)
- Pointers & `malloc` free-list intuition first; that is what makes arrays click.
- Week 3: implement a singly linked list from scratch, no STL, then reverse it
  iteratively **and** recursively (classic interview question).
- Week 5: sorting — write bubble, insertion, merge, quick; then measure on
  100k ints. Merge sort `O(n log n)` stable, quick `O(n²)` worst case.
- Complexity table to memorise:

| Structure | Access | Search | Insert | Delete |
|-----------|--------|--------|--------|--------|
| Array     | O(1)   | O(n)   | O(n)   | O(n)   |
| Linked list | O(n) | O(n)   | O(1)*  | O(1)*  |
| Hash map  | —      | O(1) avg / O(n) worst | O(1) | O(1) |
| BST       | O(log n) | O(log n) | O(log n) | O(log n) |

## Semester-1 paper pattern (2024 scheme)
- Section A: 10 × 2 marks, definitions only.
- Section B: 5 × 8 marks, one full derivation.
- Section C: 2 × 15 marks, write the code *and* trace it on paper.
- Viva: 20 marks — they always ask "difference between structure and class"
  and "why is Java platform independent (bytecode + JVM)".

## JARVIS study prompts to use nightly
- "Jarvis, quiz me on German dative from my notes."
- "Jarvis, read my CS notes on binary search trees."
- "Jarvis, take a note: I struggled with modular inverse today."
