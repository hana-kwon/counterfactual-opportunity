# Data

## What is in this folder

`hypothetical_firms/` holds the cached answers of three models to the hypothetical firm pairs (Claude Sonnet 5.5, model id `claude-sonnet-5-5`; Claude Haiku 5.5, `claude-haiku-5-5`; Qwen2.5-3B-Instruct, Hugging Face revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, greedy decoding). The firm pairs are invented by the authors and rebuilt by `src/01_hypothetical_firms.py`; each answer carries the SHA-256 of its prompt, and step 01 checks that every rebuilt prompt matches.

`model_outputs/` holds the cached answers of Claude Sonnet 5.5 for the two randomized experiments, one answer per prompt. They are keyed only by the identifiers that the public replication packages already use (`uid` for YouWiN!, `client_key` for the Egyptian experiment) and contain no applicant or borrower data. The filled prompts are not shipped, because they contain fields from the replication data; the templates and builders are in `src/prompts_grant.py` and `src/prompts_lending.py`, and the full path rebuilds every prompt and checks its hash against these files.

- `youwin_answers_claude-sonnet-5-5.csv`: 7,364 rows = 1,841 applicants x 2 information sets (`X` application only, `XR` with financing facts) x 2 tasks (`F` probability of outside financing without the grant, `LA` employment in three years without and with the grant).
- `egypt_answers_claude-sonnet-5-5.csv`: 7,028 rows = 1,004 clients x 7 conditions (`P0`, `P_R`, `P_B`, `P_RB`, `P_RBP` profit-gain objective with growing information; `S_RBP` repayment objective; `L_RBP` profit-level objective).

`inputs_manifest.json` lists the SHA-256 of every shipped file and of the replication files used for the reference run.

## What you must download (full path only)

The replication data are not redistributed here. Download both packages yourself under their terms and unzip them anywhere inside `data/raw/`; the code finds the files by name.

1. YouWiN! business plan competition, McKenzie (2017, American Economic Review 107(8)). openICPSR project 113101, version V1, "Replication data for: Identifying and Spurring High-Growth Entrepreneurship: Experimental Evidence from a Business Plan Competition", https://doi.org/10.3886/E113101V1 (https://www.openicpsr.org/openicpsr/project/113101/version/V1/view). Files used: `data/BaselineandFirstFollowup.dta`, `data/ThirdFollowup.dta`.
2. Egyptian lending experiment, Bryan, Karlan, and Osman (2024, American Economic Review 114(9)). openICPSR project 192297, version V1, "Data and Code for: Big Loans to Small Businesses: Predicting Winners and Losers in an Entrepreneurial Lending Experiment", https://doi.org/10.3886/E192297V1 (https://www.openicpsr.org/openicpsr/project/192297/version/V1/view). The package README states a Creative Commons non-commercial license. Files used: `Clean Data/16-ABA-All-Merged.dta`, `Clean Data/20-ABA-ML-group.dta`, `Clean Data/10-ABA-baseline-clean.dta`. (The package README cites a V2 DOI; on 2026-10-31 only V1 was listed on openICPSR.)
3. Optional: the online appendix of Bryan, Karlan, and Osman (2024), from the article page (https://doi.org/10.1257/aer.20220616), saved in `data/raw/` under a name that starts with `aer_online_appendix` and ends in `.pdf`. Two of the 50 psychological statements are cut at 80 characters in the Stata variable labels; with the appendix and `pdftotext` (Poppler) installed, step 04 restores their full wording and all 7,028 prompt hashes match. Without it, the three conditions that include the statements do not match by hash; the analysis does not depend on this check.

Example layout:

```
data/raw/113101-V1/data/BaselineandFirstFollowup.dta
data/raw/113101-V1/data/ThirdFollowup.dta
data/raw/192297-V1/Clean Data/16-ABA-All-Merged.dta
data/raw/192297-V1/Clean Data/20-ABA-ML-group.dta
data/raw/192297-V1/Clean Data/10-ABA-baseline-clean.dta
data/raw/aer_online_appendix.pdf            (optional)
```

Then run `python run_all.py --full`. Row-level intermediate files are written only to `outputs/<run>/work/`, which is ignored by Git.

## Not included

The 200 real startup descriptions used for the response-format check come from two commercial databases (PitchBook and Stears) and cannot be shared. The repository demonstrates the same evaluation logic on the hypothetical firm pairs and on synthetic experiments.
