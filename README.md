# Counterfactual Opportunity: Can AI Evaluators of Ventures Target Added Value?

Research artifact for the Organization Science "AI-Enabled Frontiers" Stage 1 proposal (study short name: Grant Lottery).

## What the prototype shows

Allocation decisions require judging which firms would gain most from financing, not which firms will do best. The prototype asks a language model to forecast each firm's outcome with and without financing, ranks firms by the forecast gain (or by other criteria), and scores those rankings against randomized experiments, where the realized effect of financing is identified. It varies what the model is told (information sets) and what it is asked to pursue (objectives) while holding the firm fixed.

The repository contains three parts that share one evaluation library (`src/evallib.py`):

1. Hypothetical firm pairs with a known answer key (18 pairs, two orders, two formats, four information sets). Cached answers of Claude Sonnet 5.5, Claude Haiku 5.5 and the open-weight Qwen2.5-3B-Instruct are scored against the key.
2. The YouWiN! grant lottery in Nigeria (McKenzie 2017): 1,841 applicants, grants allocated by lottery within strata, employment about three years later. Claude Sonnet 5.5 read each application and forecast employment with and without a grant.
3. A larger-loan experiment in Egypt (Bryan, Karlan, and Osman 2024): 1,004 microfinance clients randomly offered a loan about four or two times their previous loan. Claude Sonnet 5.5 forecast acceptance, late payment and monthly profit under both offers, under seven prompt conditions (7,028 answers).

The evaluation code computes (a) the policy value of allocating by a score relative to random allocation, holding the number treated fixed within each randomization stratum or loan-officer group, with ties averaged; (b) treatment x score interaction regressions with stratum or officer fixed effects; (c) contrasts of interaction coefficients across prompt conditions with standard errors clustered by randomization block and a Bonferroni adjustment for the four pre-specified comparisons; and (d) the expected overlap of the clients selected under two prompt conditions.

## How to run

Requirements: Python 3.9 or later and the packages in `requirements.txt` (`pip install -r requirements.txt`). Everything runs on a laptop CPU. No model is called; all model answers are cached in `data/`.

Demo path (no restricted data, about 5 minutes):

```
python run_all.py --demo
```

It scores the cached answers on the hypothetical firm pairs, generates two synthetic randomized experiments with known heterogeneous effects (a stratified grant lottery and a lending experiment with loan-officer blocks), creates synthetic evaluator answers in the exact formats of the real model answers, and runs the same evaluation steps as the full path. Because the synthetic data store both potential outcomes, every estimate is reported next to its true value. A final step redraws both synthetic experiments 200 times to check bias and interval coverage.

Full path (reproduces the proposal numbers, about 1 minute after the data are in place):

```
python run_all.py --full
```

It needs the two public replication packages in `data/raw/`; see `data/README.md` for exact sources, file names and checksums.

Each run writes to a new folder `outputs/<run>/` (the script refuses a non-empty folder) and ends with `outputs/<run>/receipt.json`, which records the SHA-256 of every input and output, package versions and step times. Row-level intermediate files go to `outputs/<run>/work/` and are never committed.

## Repository layout

```
run_all.py                     runs the steps for --demo or --full and writes the receipt
config/params.json             seeds, bootstrap draws, synthetic design sizes
config/proposal_numbers.json   numbers printed in the proposal, used by the comparison step
src/evallib.py                 shared evaluation code (policy value, bootstrap, interactions, contrasts, overlap)
src/prompts_grant.py           prompt templates and builder for the grant competition
src/prompts_lending.py         prompt templates and builder for the lending experiment
src/01_hypothetical_firms.py   builds pairs, prompts and answer key; checks prompt hashes; scores cached answers
src/02_synthetic_experiments.py   demo: synthetic lottery and lending experiment with known effects
src/03_youwin_build.py         full: links the YouWiN! files, rebuilds outcomes and strata, checks prompts
src/04_egypt_build.py          full: lending design rows and the original study's ML benchmark; checks prompts
src/05_grant_policy_value.py   allocation by score vs lottery, paired stratified bootstrap
src/06_grant_heterogeneity.py  grant x score interactions (McKenzie 2017, Appendix Table A18 form)
src/07_lending_evaluation.py   four pre-specified comparisons, policy values, forecasts, overlap
src/08_simulation_check.py     demo: bias and coverage over 200 synthetic draws
src/09_figures.py              figures from the aggregate outputs
src/10_compare_reference.py    full: proposal numbers; demo: estimates vs truth
data/                          cached model answers and data/README.md
reference/full, reference/demo aggregate outputs of the reference runs (no row-level data)
```

## Results of the reference runs

All numbers below are read from files in `reference/`, which were produced by `run_all.py` in this repository.

### Full path (`reference/full/`)

The full path reproduces every number it covers to the precision printed in the proposal and summary; the largest absolute difference from the authors' original analysis run is 5.5e-12 (`reference/full/10_compare_reference/comparison.csv`). All 7,364 YouWiN! prompts and all 7,028 Egypt prompts rebuilt from the replication data match the hashes stored with the cached answers (`03_youwin_build/prompt_hash_check.json`, `04_egypt_build/prompt_hash_check.json`; the Egypt check used the optional appendix PDF). The link to the YouWiN! data reproduces the employment effects of McKenzie (2017, Table 4): 5.225 for new and 4.391 for existing applicants (`03_youwin_build/aer_table4_check.csv`).

Hypothetical firm pairs (`01_hypothetical_firms/information_accuracy.csv`): with both the investment effect and the financing alternative stated, Sonnet answers correctly in 100 percent of 72 cases, Haiku in 97 percent and Qwen2.5-3B in 42 percent. With the investment effect but not the financing alternative, Sonnet withholds judgment in 89 percent and Haiku in 24 percent.

YouWiN! allocation (`05_grant_policy_value/policy_contrasts.csv`): the main AI rule (largest forecast added employment, financing facts included) yields 6.35 jobs per applicant against 6.26 under the lottery, a difference of 0.09 (paired bootstrap 95 percent interval -0.37 to 0.55; minimum detectable difference 0.65). No rule's interval excludes zero. Tie-breaks settle 27 percent of the main rule's places.

YouWiN! heterogeneity, exploratory (`06_grant_heterogeneity/`): among 461 existing businesses, the grant effect on employment is 2.30 jobs larger per standard deviation of the AI score (95 percent interval 0.61 to 4.00); the judges' score coefficient is -0.31 (-1.76 to 1.14); the difference is 2.61 (paired bootstrap 0.68 to 4.77). Among 1,044 new businesses neither score separates effects (0.01 and 0.46). The AI score does not separate profit effects in either group.

Egyptian lending (`07_lending_evaluation/`): none of the four pre-specified comparisons distinguishes the conditions. Adding alternative financing changes the larger-offer x profit-gain-score interaction by -4,378 Egyptian pounds per standard deviation (98.75 percent interval -10,037 to 1,281; minimum detectable difference 7,566). The objective comparisons are 1,668 (-669 to 4,006) for profit vs repayment objective on profit-gain scores, 0.022 (-0.083 to 0.127) for repayment vs profit objective on repayment scores, and -631 (-4,931 to 3,669) for profit-gain vs profit-level objective. Giving the larger offer to the higher-scoring half of each loan officer's clients (profit-gain score, all information) raises average monthly profit by 1,230 pounds over random assignment (unadjusted 95 percent interval 133 to 2,327); with baseline facts only, 1,079 (57 to 2,100); with the original study's machine-learning predictions, which were fitted to these outcomes, 2,624 (1,586 to 3,662). The model calls the information insufficient in 86 to 99.9 percent of answers and forecasts lower acceptance of the larger offer (59 to 64 percent against 78 to 81 percent); its forecast change in late payment ranges from -0.8 to 1.6 percentage points, whereas the measured change in the share with a perfect repayment record is -12.9 points (SE 3.2). When only the objective sentence changes, the expected overlap of selected clients is 0.80 for profit-gain forecasts and 0.51 for repayment forecasts, against 0.97 for identical and 0.50 for random rankings (`07_lending_evaluation/overlap.csv`).

### Demo path (`reference/demo/`)

The hypothetical-firm results are the same as in the full path, because they use the same cached answers (`01_hypothetical_firms/`).

Synthetic grant lottery (1,800 applicants, 24 strata, 722 winners; `02_synthetic/design.json`). The true employment effect is large only for financially constrained firms in high-opportunity sectors. The synthetic "application only" evaluator cannot see the financing facts; the "with financing facts" evaluator can. Allocating by the second evaluator's forecast added employment gains 0.53 jobs per applicant over the lottery (paired bootstrap 95 percent interval 0.21 to 0.82; true gain 0.43). The judges' score, which tracks firm size rather than the effect, gains 0.23 (-0.10 to 0.56; true gain 0.01). Among existing firms the grant x AI-score interaction is 1.46 per standard deviation (0.80 to 2.13; true slope 1.70) and the judges' score interaction is -0.06 (-0.74 to 0.61; true slope 0.06); no score separates the constant profit effect. All 8 policy contrasts and all 8 interaction intervals cover their true values (`05_grant_policy_value/policy_contrasts.csv`, `06_grant_heterogeneity/interactions.csv`).

Synthetic lending experiment (1,000 clients, 160 loan officers, 306 randomization blocks). The true profit gain is large only for credit-constrained owners who plan to buy equipment or inputs. In this draw, allocation by the all-information profit-gain forecast gains 838 per month over random assignment (95 percent interval 422 to 1,255) against a true gain of 360, so this interval misses the truth; the baseline-facts forecast gains 197 (-262 to 655; true 1). Five of the seven lending policy intervals cover the truth (`07_lending_evaluation/policy_values.csv`). One draw cannot judge an estimator, so step 08 redraws both experiments 200 times (`08_simulation_check/summary.json`): the grant policy contrast has mean bias -0.014 jobs (Monte Carlo SE 0.011) against a mean true gain of 0.41; the grant x AI-score interval covers the true slope in 96.5 percent of draws; the lending policy contrast has mean bias 36 (Monte Carlo SE 15) against a mean true gain of 352, and its 95 percent interval covers the truth in 95.5 percent of draws.

## Proposal numbers this repository does not reproduce

The following numbers in the proposal or summary come from the authors' project analyses and are not recomputed here:

- the 200 real startup descriptions (both models scored all 200 while calling every description insufficient, and declined for every firm when allowed): the inputs are licensed commercial data;
- Lee (2009) bounds on the YouWiN! main rule (-0.77 to 0.78), the response-reweighting check, the gain among existing applicants only (0.91 jobs), and the re-ask with decimal answers (+0.40): these are supporting or exploratory analyses not ported;
- the statistical-model row of the proposal's YouWiN! figure refits the model in every bootstrap replication; this repository reports the cross-fitted model without refitting (0.48, -0.02 to 1.05), so that row and the decimal-answer row of the repository's figure differ from the proposal figure;
- the measured take-up of the larger and standard offers (86 against 76 percent) and the planned minimum detectable difference (3,337), which come from the original study's Stata tables and the pre-analysis power calculation;
- the loan officers' recorded forecasts as a benchmark, and the check that the model recognizes the published Egyptian study.

## What the prototype does not do

It does not call a language model by default; the cached answers were collected with one call per prompt, so the Egypt comparisons cannot separate the effect of the objective from response variation. It evaluates retrospective rankings; it does not show that loan officers or program staff would decide better with AI advice. The randomized datasets are not redistributed. The model recognized the published Egyptian study when asked only about the setting, which does not establish that it memorized individual records.

## Citation

If you use this repository, please cite the Organization Science proposal "Counterfactual Opportunity: Can AI Evaluators of Ventures Target Added Value?" and the two source studies:

- McKenzie, D. (2017). Identifying and spurring high-growth entrepreneurship: Experimental evidence from a business plan competition. American Economic Review 107(8), 2278-2307. Data: openICPSR 113101 V1.
- Bryan, G., D. Karlan, and A. Osman (2024). Big loans to small businesses: Predicting winners and losers in an entrepreneurial lending experiment. American Economic Review 114(9), 2825-2860. Data: openICPSR 192297 V1.

## License

To be chosen by the authors. The cached model answers are keyed to identifiers of the public replication packages; use of those packages is governed by their own terms.
