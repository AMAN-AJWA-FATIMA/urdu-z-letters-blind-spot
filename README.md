# Urdu "Z" Letters: A Blind Spot in a Small Open-Weight Language Model

**Author:** Aman Ajwa Fatima, Department of CSIT, NED University of Engineering & Technology, Karachi
**Submission for:** Fatima Institute technical challenge, "Blind Spots of Frontier Models"
**Model evaluated:** [`openbmb/MiniCPM5-2B`](https://huggingface.co/openbmb/MiniCPM5-2B) (Hugging Face)

> **Summary:** I tested whether a small open-weight model can spell 48 common Urdu words with the correct one of the four "z" letters (ز, ذ, ض, ظ) when given only the Roman Urdu form. The model wrote ز in 45 of its 48 outputs and spelled none of the 36 words that require ذ, ض or ظ correctly. It treats four distinct Urdu letters as one sound.

### Key findings

| # | Finding |
|---|---|
| 1 | **Always the same letter.** Of 48 outputs, 45 contained ز. ذ, ض and ظ were never produced, not even once. |
| 2 | **0 / 36** words that need ذ, ض or ظ were spelled correctly. |
| 3 | **At the level of "always guess ز".** Writing ز every time would score 12 / 48 (25%) on the letter choice. The model scored 11 / 48 (22.9%). |
| 4 | **Even the "easy" letter is shaky.** For ز words the letter was right 11 / 12 times, but only 6 / 12 words were spelled exactly right. |
| 5 | **Script drift.** 4 of 48 outputs contained Hindi (Devanagari) letters: 2 fully Devanagari and 2 mixed with Urdu script. |

---

## 1. The blind spot (Question 1)

As a native Urdu writer, I know that correct Urdu spelling requires more than knowing how a word sounds. Urdu has four letters (ز, ذ, ض and ظ) that are pronounced similarly but used in different words. Choosing the correct letter often depends on knowing the word's origin and established spelling, something native writers learn through reading and practice.

I have noticed that AI tools can produce Urdu text that appears fluent at first but contains incorrect vocabulary or spelling. One particular problem is confusing words that sound alike but are written with different letters. Such errors can be easy to overlook, especially for students who use AI to prepare assignments, essays, or formal documents. Another issue I have encountered with Urdu voice input is the possibility of speech being transcribed into Hindi using Devanagari script rather than Urdu script. This makes the output unsuitable for someone who needs correctly written Urdu.

Both problems showed up directly in my own test. Asked to write the Roman Urdu word *zulm*, the model produced زولم instead of ظلم; for *zikr* it returned a Hindi-script mix (जकر) instead of ذکر. The Hindi-script problem appeared in 4 of my 48 outputs.

These problems matter in Pakistan, where students and professionals increasingly rely on AI for writing, learning, translation, and communication. Incorrect Urdu can affect the clarity, credibility, and cultural authenticity of their work. It can also make language technology less useful for people who prefer to communicate in their native language.

I believe standard AI benchmarks can overlook this weakness because they often prioritize meaning, fluency, or translation quality. A sentence may preserve its intended meaning even when a word contains the wrong Urdu letter. Therefore, evaluating Urdu AI requires specific tests of orthographic accuracy, including whether models consistently distinguish ز, ذ, ض and ظ in everyday vocabulary.

---

## 2. What was tested

### 2.1 Question
Given a common Urdu word written in Roman Urdu (Latin letters), can the model write it in Urdu script with the **correct one of the four "z" letters**?

### 2.2 Dataset (`words.csv`)
- **48 words**, hand-written by the author, a native Urdu speaker.
- **12 words per letter** (ز, ذ, ض, ظ), so each letter is equally represented.
- Columns: `word` (correct Urdu spelling), `correct_letter`, `roman_urdu`.
- Every Roman Urdu form contains "z", so the Roman Urdu input alone does not reveal which Urdu letter is correct. The model has to know the word.

### 2.3 Model
- **`openbmb/MiniCPM5-2B`**, a recent, widely downloaded, instruction-tuned causal language model that falls within the 0.6B-6B range required by the challenge.
- Chosen because it is small enough to run on a free Colab T4 GPU while still being a current, widely used model, so the behavior I measure is relevant to how small modern models behave in practice.
- I did not assume the model supports Urdu. A failure is informative either way: if Urdu is not a focus language, the result shows a gap in coverage; if it is, the result shows a gap that standard evaluations do not reveal.
- It is a **reasoning model**: it writes a `<think> ... </think>` section before its answer. This mattered in practice (see 2.5).

### 2.4 Procedure
- Platform: Google Colab, T4 GPU.
- Prompt (identical for every word, one word per request):
  `Write the Roman Urdu word '{roman_urdu}' in Urdu script. Reply with only the word.`
- Greedy decoding (`do_sample=False`), so the run is deterministic.
- Maximum new tokens: 1024.
- Only the text after `</think>` was kept as the answer:

```python
full = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
answer = full.split("</think>")[-1].strip()
```

### 2.5 First run (documented for transparency)
My first run used `max_new_tokens=60`. Because MiniCPM5-2B reasons before answering, every output was cut off inside the `<think>` block and none contained an answer. I increased the limit to 1024 and re-ran. The truncated first run is included as `results_truncated.csv`.

### 2.6 Scoring
Each of the 48 outputs was read and checked against the correct spelling by the author, a native Urdu speaker:

| Column | Meaning |
|---|---|
| `letters_used` | which of ز ذ ض ظ the model wrote (`none` if it wrote none of them) |
| `z_letter_correct` | 1 if the model used the correct one of the four letters, else 0 |
| `correct` | 1 only if the **whole word** exactly matches the correct spelling, else 0 (stricter) |
| `script` | `arabic` (Urdu script), `devanagari` (Hindi), or `mixed` |
| `note` | short description of what happened |

**Use of AI assistance (disclosure).** I used an AI assistant (Claude) for technical setup help (the Colab script), for a first mechanical pass of the `letters_used`, `z_letter_correct`, `script` and `note` columns (kept as `results_draft.csv`), and for formatting and proofreading of this README. I reviewed every row, made the final scoring decisions, scored the `correct` column myself, and wrote the observations and conclusions. The word list is entirely my own.

---

## 3. Results

### 3.1 Accuracy per letter

| Correct letter | Used the right "z" letter | Whole word exactly right |
|---|---|---|
| ز | 11 / 12 (91.7%) | 6 / 12 (50.0%) |
| ذ | 0 / 12 (0%) | 0 / 12 (0%) |
| ض | 0 / 12 (0%) | 0 / 12 (0%) |
| ظ | 0 / 12 (0%) | 0 / 12 (0%) |
| **All** | **11 / 48 (22.9%)** | **6 / 48 (12.5%)** |

### 3.2 Which letter the model actually used (confusion table)

| Correct letter ↓ / Model wrote → | ز | ذ | ض | ظ | none of the four |
|---|---|---|---|---|---|
| ز | 11 | 0 | 0 | 0 | 1 |
| ذ | 11 | 0 | 0 | 0 | 1 |
| ض | 11 | 0 | 0 | 0 | 1 |
| ظ | 12 | 0 | 0 | 0 | 0 |
| **Total** | **45** | **0** | **0** | **0** | **3** |

The model wrote **ز in 45 of 48 outputs**. It never wrote ذ, ض or ظ. The 3 "none" cases were outputs in Hindi script.

### 3.3 Output script

| Script | Count |
|---|---|
| Urdu (Arabic script) | 44 |
| Devanagari (Hindi) | 2 |
| Mixed Urdu + Devanagari | 2 |

### 3.4 Examples

| Roman Urdu | Correct | Model output | What happened |
|---|---|---|---|
| zameen | زمین | زمین | exact match |
| zubaan | زبان | زوبان | correct letter (ز), but an extra letter was added |
| zara | ذرا | زرا | wrote ز instead of ذ |
| zaroorat | ضرورت | زرورت | wrote ز instead of ض |
| zulm | ظلم | زولم | wrote ز instead of ظ |
| zikr | ذکر | जकر | Hindi (Devanagari) letters instead of Urdu |
| ziafat | ضیافت | ज़िफ़त | fully in Hindi script |

### 3.5 What I observed

The results suggest that the model resolves the ambiguous "z" sound by defaulting to ز, in line with Roman Urdu pronunciation rather than the original Urdu spelling. Every output that contained any of the four letters used ز: 45 of 48 outputs. None of the 36 words that require ذ, ض or ظ were spelled correctly (0 out of 36). The model also did not reliably keep Urdu script: 2 outputs were in Devanagari and 2 mixed Urdu and Devanagari. Even for ز words, where the letter choice was right 11 out of 12 times, only 6 out of 12 were spelled exactly right, with extra or changed letters elsewhere in the word. This suggests that producing understandable-looking text does not necessarily mean the model understands Urdu orthographic distinctions, though a single model and prompt cannot show why.

### 3.6 Limitations
- One model, one prompt, 48 words, one deterministic run. This is a small pilot, not a full benchmark.
- The Roman Urdu input is ambiguous (four letters share the sound "z"), so the test cannot fully separate "the model does not know the spelling" from "the model defaults to ز when unsure".
- Scoring was done by one person (the author), and some Roman Urdu spellings can correspond to more than one Urdu word.
- Exact-match scoring (`correct`) is strict: a spelling difference may sometimes be an acceptable variant. For this reason the letter-level column (`z_letter_correct`) is reported alongside it.
- Words are everyday vocabulary; rarer words were not tested separately.

---

## 4. Path forward (Question 3)

I believe the most effective solution is to combine carefully curated Urdu spelling data with targeted supervised fine-tuning and spelling-level evaluation. My results suggest that the model relies heavily on Roman Urdu pronunciation, mapping the "z" sound to ز instead of identifying the correct spelling of each word.

To address this, I would develop a high-quality dataset containing common Urdu words written in their correct script, paired with their Roman Urdu forms and correct-letter labels. The dataset should include balanced examples of ز, ذ, ض and ظ, with words appearing in different contexts to help the model learn established spellings rather than simply memorizing isolated examples.

I would then use parameter-efficient fine-tuning, such as LoRA, to teach an open-weight model these distinctions without retraining the entire model from scratch. To evaluate the improvement, I would first repeat the original 48-word test and compare letter-level accuracy, exact word accuracy, and script consistency before and after fine-tuning. The 48 test words would be kept out of the training data so that gains cannot come from memorization. I would also introduce unseen words and sentences to test whether the model generalizes beyond the training examples. Finally, I would include these tests in a dedicated Urdu orthographic benchmark so that future evaluations measure spelling accuracy alongside fluency and meaning.

I consider this approach promising because the observed failure was consistent in this pilot rather than random: the model repeatedly favored ز for every "z" sound. Explicit spelling supervision could help it distinguish pronunciation from orthographic convention. However, improvement on unseen words would be essential evidence that the model has learned a more reliable spelling pattern rather than merely memorized the dataset.

---

## 5. Repository contents

| File | Description |
|---|---|
| `README.md` | this document |
| `words.csv` | the 48-word evaluation dataset |
| `results_truncated.csv` | first run, cut off during the model's reasoning (no usable answers) |
| `results.csv` | second run, raw model outputs |
| `results_draft.csv` | first mechanical draft of the scoring columns, before manual review |
| `scored.csv` | final hand-checked scores (`letters_used`, `z_letter_correct`, `correct`, `script`, `note`) |
| `notebook.ipynb` | the Colab notebook used to run the model |

## 6. How to reproduce

1. Open the notebook in Google Colab and select a T4 GPU runtime.
2. Upload `words.csv`.
3. Run all cells. The notebook loads `openbmb/MiniCPM5-2B`, runs the prompt above on each word, removes the `<think>` section, and writes `results.csv`.
4. Score the outputs by comparing each to `word` and filling the columns in section 2.6.

## 7. Acknowledgements
Challenge and evaluation design by Fatima Institute. Model: `openbmb/MiniCPM5-2B`. Word list, final scoring decisions, observations and conclusions: the author.
