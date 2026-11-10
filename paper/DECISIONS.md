# DECISIONS AND STYLE GUIDE (binding for every writer, reviewer and auditor)

Last updated by the controller after the user's instruction "resume the agents and continue to writing until it finishes".
Items marked PROVISIONAL are controller defaults adopted so writing can proceed; the user may change them, and then the affected sections are rewritten.

## 1. Decisions made by the user
- D1. Central claim: option (c), the paper is re-framed around the decoder and the benchmark finding, and the missing ablations are run first (NOMP+self-calibration done; retrained no-IABC network and multi-seed need a GPU = not available).
- D2. The learned unrolled decoder is NOT part of this paper. Use the decoder exactly as it is in `IABR_v2_extracted\IABR_v2\notebook_src.py` section 7 (lines 527-647) and its existing results. Nothing from `evidence\learned_decoder\` training runs may be cited (the diagnostic files `diag*` are allowed as analysis of the classical decoder, with the caveat in PAPER_ARCHITECTURE.md).
- D3. The decoder is an adaptation of Newtonized OMP (NOMP). NOMP must be cited and the network-free decoder must appear in every main results table.

## 2. Provisional defaults (controller)
- P-A. Framing: a re-evaluation / strong-baseline paper ("what does the network contribute?"). Not a method-novelty paper.
- P-B. Article type: regular paper (about 10-12 two-column pages). Authors, affiliations, corresponding author, funding, journal: write `[REQUIRES USER INPUT]`.
- P-C. Headline metric: strict Pd primary; paper-style Pd always shown beside it (comparability with Lloria et al. and Meneses-Albala et al.). RMSE is conditional on detection and is always labelled so. Do not mix these with the component-wise assignment score of the diagnostics (spec 10.2).
- P-D. Edge / Jetson / real-time / energy claims: removed from the paper. Only measured or analytic items may appear, with the machine named.
- P-E. Names: "network-free decoder" (the NOMP-type decoder with greedy FFT initialisation, no network); "NOMP+selfcal" (that decoder plus the self-calibration stage); "hybrid (IABR-v2)" (CNN + decoder); "IABC" for the learned per-antenna corrector (expansion `[REQUIRES USER INPUT]`); "decoder on the uncorrected channel" instead of "v2 no IABC"; "U-Net of [TVT]" for the baseline. Do not write "NOMP" for the in-house decoder without saying it is an in-house adaptation (N12 of PAPER_ARCHITECTURE.md).
- P-F. Working title: "How Much Does the Network Contribute? A Newtonized Matching-Pursuit Decoder with Array Self-Calibration as a Strong Baseline for Joint AoA/AoD Estimation in Analog mmWave MIMO" (architecture section 1.1). The user chooses the final title.

## 3. The novelty audit overrides the architecture (NOVELTY_TABLE.md)
- Contribution P2 as "CNN-initialised Newton/NOMP refiner" is NOT novel (Li et al. TWC 2025; Schieler et al. 2025; Barthelme and Utschick 2021): present the hybrid only as the compared model and as an object of decomposition, and cite the prior art in Related Work.
- Self-calibration (P3) is prior art in its components (Arai and Ishibashi, TVT 2025, and others): present the self-calibration stage as an adaptation to the 2-D beam-space model, credited; the contribution is the evaluation, not the invention.
- Identifiability (P7) is known for ULAs: it belongs in background/appendix with citations, never as a contribution.
- "To the best of our knowledge" may be used only for P1 (a Newton-refined classical decoder evaluated on the Lloria/Roger benchmark) and P4, scoped to the benchmark, and only once. Forbidden everywhere: first, novel, state-of-the-art, outperforms (use "scores higher than ... on ..."), edge-ready, real-time, robust (unless the exact tested condition is named).
- Prior-art works named in NOVELTY_TABLE.md but not in `citations\topic_*_candidates.md` (Li et al. TWC 2025, Schieler et al. 2025, Barthelme and Utschick 2021, BT-NOMP, NeuPLSE, others) have NOT been through the citation audit. Cite them with the tag `[NV:<short name>]` (e.g. `[NV:Li2025]`); the bibliography builder will verify them before the end. Never invent bibliographic details for them.

## 4. Citation policy
- Cite audited records as `[C1-001]`, `[C2-003]`, ... exactly as in `citations\topic_*_candidates.md`, applying the corrections in `citations\audit_*.md` (reworded claims, corrected authors, DOIs). If an audit verdict is FAIL for a claim, do not use that claim. Duplicate records (listed in PAPER_ARCHITECTURE.md section 3 II) are cited by their first id.
- Anything else that needs a reference but has no record: write `[CITATION NEEDED: <claim>]`. Never invent a DOI, venue, year, pages.
- Quantitative statements from `citations\topic_5_candidates.md` may be used only if marked SUPPORTED in `citations\audit_5.md` (the 16-element half-power beamwidth 6.36 degrees is a derived fact: give the formula).

## 5. Numbers policy
- Every number in the text must be traceable. After each sentence or table row that contains a result, add an HTML comment `<!-- src: <file>, <section or line or table> -->`. The LaTeX stage strips the comments, the experiment auditor uses them. A number without a source is an error.
- Use the paper-style Pd values of RESULTS.md and the new results in `evidence\new_results\` (when they exist). Use only numbers that exist in the evidence files; if a number is missing, write `[NOT YET RUN]` or `[REQUIRES USER INPUT]`.
- Always name: bank, metric, whether L is given, and the interval when one exists. Single trained network seed and single data draws are stated where results are discussed.

## 6. Writing style
- IEEE journal register, third person plural ("we evaluate"), present tense for the method, past tense for experiments. Short, concrete sentences. Define every symbol at first use. State every assumption and what breaks if it fails.
- No marketing language, no filler transitions (avoid "Furthermore", "Moreover", "Notably", "It is worth noting"), no rhetorical questions, no bullet lists in the body except where an IEEE paper would use one (contributions, assumptions, algorithm steps).
- Format of section files: Markdown, math in LaTeX (`$...$`, `$$...$$`) with equation labels written as `\label{eq:...}` inside a `$$` block; figures and tables are not drawn in the section file: put a block `FLOAT: Fig. N` / `FLOAT: Table N` at the end of the file with caption text, data source, and status (existing file, to redraw, or new).
- Sign and notation conventions come from `MASTER_RESEARCH_SPEC.md` section 6 and are used consistently: do not change symbols between sections.

## 7. Open `[REQUIRES USER INPUT]` list (do not fill)
Affiliations, corresponding author, funding (authors' names are fixed in section 9); journal; IABC expansion; code and data availability, weights licence, DLDOA commit hash and `.h5` location; dldoa_test files; GPU runs (retrained no-IABC network, >= 3 seeds, impairment-trained U-Net); Jetson measurement (removed by default); hardware details of the v1 run machine.

## 8. New evidence (experiment runner, evidence\new_results\RESULTS_NEW.md) that overrides earlier statements
- The +0.046 gain of self-calibration at 5 degrees separation (0.8970 -> 0.9430) is NOT calibration: selfcal() with lam = 0 (no correction) still runs 2 outer iterations x cyclic(rc=2) = 4 extra cyclic rounds and scores 0.9595; plain greedy with rc = 8 scores 0.9640; the calibration part itself lowers Pd by 0.0165 [0.009, 0.025] on that bank. State this: the default of 3 cyclic rounds is under-converged for close paths; self-calibration helps only under gain error (gain 4 dB: 0.8697 -> 0.8950).
- Diagnostics re-run on the network-free decoder (2000 official scenes, component-wise score, NOT paper-style Pd): greedy 0.7548, selfcal 0.7542; 98 percent of failing scenes fit the data at least as well as the truth (1,100 of 1,120; of the other 20, 6 are inside the likelihood-ratio band and 14 are optimiser misses with the corrected statistic 2*dE; the superseded diag2 convention gave 8 and 12 and appears only as a footnote row of Table VII-14); joint-Fisher CRB limit 0.798 (wrap) / 0.804 (clamp); end-fire components 819, observed 0.319 vs 0.331 / 0.409, 222 of 387 fold errors are end-fire. The earlier single-path CRB (0.802/0.807) was slightly optimistic; the likelihood-ratio statistic for complex noise is 2*dE (corrected in e5). Use these numbers, not the diag2/diag3 ones.
- Headline (official bank, paper-style Pd): U-Net 0.7366, hybrid 0.7551, network-free 0.7581, NOMP+selfcal 0.7576; differences with paired 95 percent intervals: network-free - U-Net +0.0215 [+0.0181,+0.0248]; hybrid - network-free -0.0030 [-0.0043,-0.0016]; NOMP+selfcal - hybrid +0.0025 [+0.0014,+0.0036]; NOMP+selfcal - network-free -0.0005 [-0.0014,+0.0005]. Over 51 banks (interval excludes 0): network-free - U-Net > 0 on 39, < 0 on none; hybrid - network-free < 0 on 17, > 0 on 1 (gain 4 dB / SNR 15, +0.0427). Not corrected for multiple comparisons; single data draw per bank; single network seed.
- Not processed: dldoa_test (bank file absent), nuis_* and v1_clean_* (no cached CNN outputs). Batch-1 CPU latency (VivoBook i5-1135G7, NumPy, L = 3, laptop not idle): greedy 9.96 ms (1 thread) / 10.46 ms (6 threads); greedy + selfcal 16.17 / 15.91 ms.

## 9. Authors (given by the owner on 2026-10-03)
1. Mishat Hasan Milon
2. Khalid Mehebub
3. Khondokar Asif Zaman
The numbers (1), (2), (3) were taken as the author order; the owner has not said whether they instead mean affiliation labels: confirm. Still `[REQUIRES USER INPUT]`: affiliations, corresponding author and e-mail, funding, acknowledgements, ORCIDs, competing-interest statement. In the manuscript use exactly these names in this order in the title block; do not invent affiliations.

## 10. Owner's rewrite instructions (2026-10-03) and how they were applied
- Title fixed by the owner: "IABR-Net: An Impairment-Adaptive Network for Joint Estimation of AoA and AoD from mmWave". Target: IEEE Transactions, at most 15 pages, readable, only what a Q1 paper needs. NN (IABR-Net) is the focus.
- Method as reported: CNN heatmap -> heatmap peaks seed the refinement (NO greedy FFT path, NO dual initialisation) -> Newton + cyclic refinement -> self-calibration, on the IABC-corrected channel. Main results are therefore the heatmap-seeded variant (evidence/new_results/bank_json_heat/, consistent with RESULTS.md ablation row "v2 heatmap init only").
- The owner asked to remove the results of the decoder without the network ("NOMP only"). Applied: it is gone from the main tables and figures and from the abstract/contributions. NOT applied: one row "FFT seeds, no network" stays in the ablation table (Table IV) with its sentences in Sec. VII-D, VIII, because without it the paper would imply that the network is what produces the accuracy gain, which the evidence does not show (FFT seeds give +0.0073 on ideal hardware, +0.031 at L=8, +0.415 at 5 degrees separation). The owner may disagree; a successor must not delete it silently.
- Reported weakness that follows from heatmap-only seeding: two paths 5 degrees apart: 0.528 vs 0.869 (U-Net).
- New files: paper/latex_v2/ (main.tex, tables from make_tables.py, figures from figs/make_figs.py and figs/fig2.py), PDF paper/IABR-Net_paper_v2.pdf, Overleaf zip paper/latex_v2_overleaf.zip. The earlier long draft (paper/latex/, sections/*.md) is kept as the superseded version.
- Sweeps for gain/phase/separation with the heatmap-only method were not computed (the owner stopped further decoding runs); the robustness figure uses the banks that exist: L sweep, gain 2 dB at 0 dB, gain 4 dB at 15 dB, phase 15 degrees at 15 dB, and 5 degrees separation (text/table).

## 11. Author details and acknowledgment (owner, 2026-10-03)
Order: 1 Mishat Hasan Milon (2106059@eee.buet.ac.bd), 2 Khalid Mehebub (2106065@eee.buet.ac.bd), 3 Khondokar Asif Zaman (student ID given as "21060660", which has 8 digits: confirm before writing the e-mail). Affiliation written from the e-mail domain: Department of Electrical and Electronic Engineering, BUET, Dhaka (inferred; confirm). Acknowledgment: Ashik Abrar Naeem, who selected and mentored the project in the AI/ML course. Funding, competing interests, code/data availability still [REQUIRES USER INPUT]. A Key-terms table (paper-style Pd, strict Pd, RMSE, ...) opens the paper (Table I) so that the metric names are defined before use.

## 12. Confirmed by the owner (2026-10-03)
Khondokar Asif Zaman: 2106060@eee.buet.ac.bd. Affiliation (EEE, BUET, Dhaka) confirmed for all three authors. Funding: none (stated in the paper). Still open: competing-interests statement and code/data availability.

## 13. Deep research of 2026-10-15 (NOMP vs network; IABC; ADC) - read before writing the next paper version
All verified results are collected in `paper/evidence/deep_research/RESULTS_SUMMARY.md` (sections 1-10, file index included). Key decisions: (a) on ideal hardware NOMP-only (0.7581) beats every network variant - never claim the network is needed there, and keep NOMP (Mamandipoor et al. 2016) as a named, credited baseline; (b) the old IABC is replaced by IABC-v2 (calibrates close to the CRB) but it must not be presented as essential - in the realistic error range (<= 8 deg, <= 3 dB) it gives no significant detection gain, and a classical ML self-calibration ties it in most cells; (c) the learned decoder must be described as unfolded from NOMP; (d) the most promising journal story is the low-resolution-ADC result (network read-out beats every tested classical method by +0.13 to +0.18 Pd at 1-2 bits, 15 dB), pending the quantiser-aware ML baseline in `finetune/adversarial/`; (e) the all-neural pipeline beats the published U-Net (+0.014) and ResNet (+0.040) on their benchmark. Final synthesis: `paper/evidence/deep_research/REPORT.md` (when finished).


## 14. Work of 2026-10-15 to 2026-11-10 (read before writing the next paper version)
Full details: `paper/evidence/deep_research/RESULTS_SUMMARY.md` (sections 1-14), `paper/evidence/adc_program/` (paper study notes, TAKEAWAYS.md, POSITIONING.md), `paper/evidence/audit_2026-11/` (code audit, NOMP limitations), `paper/evidence/adc_bits_test/` (1-4 bit test), `paper/explainer/ADC_Explained.pdf` and `GEMINI_SIMULATION_BRIEF.md`. Decisions: (a) the original code has no bug; the weak network-only score is a resolution/design issue (fixes listed in CODE_AUDIT.md); (b) the network helps at 1-2 bits, the hybrid wins only at 1 bit (+0.018 over quantiser ML), quantiser-aware classical ML wins at 2-4 bits, and most of the gain over NOMP comes from the exact quantiser likelihood (prior art, to be credited: Mo-Schniter-Heath 2018, GNOMP 2026); (c) GNOMP-2D baseline had a bug (scored below NOMP) and must be fixed and validated before use, EM-VAMP seeding control not yet run; (d) realistic hardware ranges are cited in RESULTS_SUMMARY.md section 14, stress tests at 20-45 deg / 6 dB are stress tests only; (e) commercial ADC bit depth not verified. The ADC program (paper/workflows/adc-program.js, run wf_e9358298-a12) is stopped after the GNOMP baseline stage.
