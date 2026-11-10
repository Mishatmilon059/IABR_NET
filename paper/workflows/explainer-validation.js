export const meta = {
  name: 'explainer-pdf-and-validation',
  description: 'Validate the full IABR-Net research (scientific soundness, literature consistency, reproducibility, hostile review), then write a non-technical illustrated explainer PDF and a journal-readiness report',
  phases: [
    { title: 'Facts', detail: 'one facts sheet: final model stage by stage, every number with its source, every citation with status' },
    { title: 'Validate', detail: 'four independent auditors: hidden flaws, literature consistency, reproducibility, hostile journal reviewer' },
    { title: 'Adjudicate', detail: 'verify every blocker, write the validated facts and the validation report' },
    { title: 'Explainer', detail: 'figures, LaTeX explainer for a non-technical reader, compile to PDF, readability and trace check' },
    { title: 'Readiness', detail: 'journal-readiness verdict and report' },
  ],
}

const DR = 'D:/ai_ml_project/paper/evidence/deep_research'
const EX = 'D:/ai_ml_project/paper/explainer'
const CTX = `
PROJECT: IABR-Net, joint AoA/AoD estimation for analog mmWave MIMO (16x16 ULAs, DFT beam codebooks), an undergraduate AI/ML course project at BUET (authors Mishat Hasan Milon, Khalid Mehebub, Khondokar Asif Zaman; mentor Ashik Abrar Naeem). Benchmark: Lloria et al., IEEE TVT 2026 (U-Net / ResNet). Classical reference: NOMP (Mamandipoor, Ramasamy, Madhow, IEEE TSP 2016); gain/phase self-calibration lineage Weiss & Friedlander 1990.
WHERE THE EVIDENCE IS (read these first): ${DR}/RESULTS_SUMMARY.md (all verified numbers, sections 1-11), ${DR}/REPORT.md (final synthesis + journal framings A/B/C), ${DR}/unfolded_v2/REPORT.md (unfolded decoder v2, if present) and every sub-folder report: verify/, stress/, stress/adversarial/, iabc/, iabc/adversarial/, iabc_v2/, iabc_v2/adversarial/, nn_decoder/, nn_decoder/adversarial/, finetune/adc/, finetune/imp/, finetune/adversarial/, nn_decoder_adapt/, unfolded_v2/*, lit/ (verified literature: baselines_survey.md, when_dl_helps.md, unfolded_v2/lit/unfolding_survey.md). Code: D:/ai_ml_project/IABR_v2_extracted/IABR_v2/notebook_src.py, iabr2_sim.py, D:/ai_ml_project/paper/evidence/new_results/common.py, the scripts in each evidence folder. Decision log: D:/ai_ml_project/paper/DECISIONS.md (section 13). Earlier paper draft: D:/ai_ml_project/paper/latex_v2/ (main.tex, references.bib).
RULES:
- Never invent a number, a citation or a statement about a paper. Every number must trace to a log/report file; every statement about prior work must be checked against the paper itself (open it: WebSearch/WebFetch via ToolSearch "select:WebSearch,WebFetch", local PDFs under D:/ai_ml_project/papers, or the verified lists in lit/). Mark anything you could not verify as UNVERIFIED instead of guessing.
- NOMP and self-calibration must be named and credited wherever they are used; the learned decoders are "unfolded from NOMP". Do not hide or soften negative results.
- Drive D: is almost full (~95 MB free): keep files on D: small (text, figures < 1 MB each, the final PDF). Put anything large in C:/Users/Milon/iabr_deep_research_cache/explainer/. Never delete anything outside your own folder. No git.
- If the Write tool refuses a .md/.tex file, write it with Bash (python or heredoc). CPU work must be light (one process, chunks <= 400 scenes) unless your task says otherwise.`

const FACTS = `${CTX}

YOUR TASK (folder ${EX}/facts): build ONE facts sheet the rest of the team will rely on: ${EX}/facts/FACTS.md (write with Bash).
1. FINAL SYSTEM, stage by stage, as it actually exists in code: input measurement (beams, Y), antenna-domain transform, IABC / IABC-v2 corrector, CNN trunk and heatmap, decoders (classical NOMP-type heat/dual decoder; learned decoder unfolded from NOMP; unfolded v2 CA-UNOMP if built), the hybrid "network seeds + physics-model ML" (quantiser ML for ADC, PN-ML for phase noise), self-calibration, outputs. For each stage: what goes in, what comes out, what is learned vs fixed, parameter count, file/line in code.
2. KEY TERMS with a one-paragraph plain definition each (AoA, AoD, mmWave, MIMO, ULA, beam/codebook, analog beamforming, SNR, ADC and bits, quantisation, AGC, per-antenna gain/phase impairment, mutual coupling, antenna position error, phase noise, near field, NOMP, OMP, Newton refinement, self-calibration, CNN, heatmap, unfolding, maximum likelihood, Cramer-Rao bound, Pd paper-style vs strict, RMSE, paired bootstrap CI, oracle, ...).
3. EVERY RESULT that may be stated anywhere: a table row per number with value, CI, condition, source file (+ log line if possible), and its adversarial status (CONFIRMED / REFUTED / NEEDS REWORDING / not checked). Include the comparisons with the published networks (U-Net, ResNet) and with NOMP.
4. EVERY CITATION used by the project (lit/ lists, latex_v2/references.bib, the reports): key, full reference, DOI/URL, verification status as recorded, and the exact claim it supports.
5. Known limitations and open items (from the reports).
Return a short summary (<= 40 lines) and the path.`

const LENSES = [
  { key: 'soundness', label: 'audit: hidden scientific flaws', prompt: `Audit for HIDDEN SCIENTIFIC FLAWS. Read the facts sheet ${EX}/facts/FACTS.md and the underlying scripts/logs. Look for: train/test leakage or overlapping seeds; test banks used for tuning, model selection or thresholds; post-hoc selection among many variants without correction; multiple comparisons; unfair or mis-configured baselines (information given to one method but not another, e.g. known noise level, known L, oracle quantities); metric misuse (paper-style Pd dropping scenes, endfire ambiguity, wrapping); simulator unrealism or bugs (impairment models, ADC model, coupling side/transpose); conclusions stronger than the evidence; numbers in reports that disagree with their logs. Re-run small checks where cheap (light CPU). For each finding: severity (BLOCKER / MAJOR / MINOR), exact location (file:line), evidence, and the fix.` },
  { key: 'literature', label: 'audit: literature consistency', prompt: `Audit LITERATURE CONSISTENCY. For every citation in ${EX}/facts/FACTS.md and every sentence in the reports that states what a paper did or found (Lloria et al. TVT 2026 and PIMRC 2024, Meneses-Albala 2026, NOMP 2016, Weiss & Friedlander 1990, SignalNet, Mo-Schniter-Heath, Li et al. 2017, Walden 1999, Barthelme & Utschick 2021, DeepFreq, SubspaceNet, mpNet, LISTA/LAMP, GNOMP, etc.): open the paper (WebFetch / arXiv / local PDF) and check (a) bibliographic metadata (authors, venue, year, volume, pages, DOI), (b) that the attributed claim is actually stated there (quote the supporting sentence, <= 15 words), (c) that the comparison with our results is fair (same benchmark, same metric, same SNR range). Flag invented or unsupported statements and wrong metadata. Also check that our use of the U-Net / ResNet numbers matches the authors' own (the ports were validated against their repository and Table II). For each finding: severity, location, evidence, fix.` },
  { key: 'reproduce', label: 'audit: reproducibility', prompt: `Audit REPRODUCIBILITY. Pick the 8-10 most important numbers in ${EX}/facts/FACTS.md (official-bank NOMP vs networks; U-Net/ResNet comparison; 1-bit ADC hybrid vs quantiser ML; position-error win; IABC-v2 removal %; unfolded v2 headline if present) and re-derive each from the saved logs/json/prediction caches with your own short script (light CPU; caches are in C:/Users/Milon/iabr_deep_research_cache/dr_cache and the evidence folders). Report: reproduced exactly / within CI / not reproducible, with your numbers. Also check that each experiment's script exists, runs (dry-run or smoke on 20 scenes) and that seeds are recorded.` },
  { key: 'reviewer', label: 'audit: hostile journal reviewer', prompt: `Act as a HOSTILE REVIEWER for IEEE Transactions on Signal Processing / Wireless Communications / Vehicular Technology. Read ${DR}/REPORT.md (framings A/B/C), ${DR}/unfolded_v2/REPORT.md if present, and ${EX}/facts/FACTS.md. For each framing write the review you would submit: summary, novelty assessment against the literature (is it new? what is the closest prior work?), technical soundness, significance, missing baselines/experiments, presentation issues, and a decision (accept / minor / major / reject) with reasons. Then state, per framing, the minimum work needed to reach "major revision or better". Be concrete and cite the evidence files.` },
]

const ADJ = (facts, audits) => `${CTX}

FACTS SHEET SUMMARY:
${facts}

AUDIT FINDINGS:
${audits}

YOUR TASK (folder ${EX}/validation): adjudicate. 1) Merge and de-duplicate all findings. 2) For every BLOCKER or MAJOR finding, independently verify it (re-read the code/logs/paper yourself; run a tiny check if needed) and mark it CONFIRMED or DISMISSED with the reason. 3) Write ${EX}/validation/VALIDATION_REPORT.md: every finding with its final status, severity, evidence, and the required fix or the limitation wording to use. 4) Write ${EX}/facts/VALIDATED_FACTS.md: the facts sheet corrected by every confirmed finding (remove or reword refuted numbers/claims, mark limitations, keep only verified citations, list UNVERIFIED items explicitly). 5) Do NOT edit the original evidence reports; corrections live only in these two files. Return the list of confirmed blockers/majors and the overall soundness verdict (<= 50 lines).`

const FIGS = (adj) => `${CTX}

VALIDATION RESULT:
${adj}

YOUR TASK (folder ${EX}/figs): make the figures for a NON-TECHNICAL explainer PDF, using ONLY numbers from ${EX}/facts/VALIDATED_FACTS.md. Matplotlib, PNG at 200 dpi (each < 600 KB), clean labels, large fonts, colour-blind-safe palette, one idea per figure. Required: (1) the problem: a base station and a phone, signal paths bouncing off buildings, AoA and AoD arrows; (2) antennas and beams: a 16-element array scanning beams like a torch; (3) the measurement grid 16x16 as a picture (heat of beam powers) for one scene; (4) ADC and bits: one wave sampled with 12, 4, 2 and 1 bit (the "ruler with fewer marks" idea); (5) hardware impairments: ideal vs per-antenna phase/gain errors, position error, phase noise (simple sketches); (6) how NOMP works: find the strongest peak, refine it (Newton), subtract, repeat - as a 3-panel cartoon; (7) the CNN heatmap: input grid -> heatmap with peaks -> starting points; (8) the full pipeline block diagram (with which blocks are learned and which are classical); (9) unfolding: an iterative algorithm drawn as stacked layers; (10) hybrid: network gives starting points, physics-model ML refines; (11) results bar charts: ideal hardware (NOMP vs our pipeline vs U-Net vs ResNet) and 1-bit/2-bit ADC (NOMP, network alone, quantiser ML, hybrid); (12) where each method wins (a simple map of conditions). Reuse existing evidence figures only if they are clear for a lay reader. Write ${EX}/figs/FIGURES.md listing each file, caption, and the data source of every number. Return the list.`

const WRITE = (adj, figs) => `${CTX}

VALIDATION RESULT:
${adj}

FIGURES:
${figs}

YOUR TASK (folder ${EX}/doc): write the explainer as LaTeX (report or article class, A4, 11pt, readable fonts) at ${EX}/doc/explainer.tex with references.bib, and compile it with D:/ai_ml_project/paper/tools/tectonic.exe (run from ${EX}/doc; tectonic.exe -X compile explainer.tex, or tectonic.exe explainer.tex) to ${EX}/IABR-Net_Explainer.pdf. Audience: a smart NON-TECHNICAL reader (e.g. a supervisor from another field or a family member). Style: plain English, short sentences, an everyday analogy and a concrete example for every concept, a figure wherever it helps, a "Key idea" box at the end of each section, no unexplained jargon (define each keyword on first use and in a glossary). Use ONLY ${EX}/facts/VALIDATED_FACTS.md for numbers and claims; cite only verified references (BibTeX from the verified lists; mark nothing as cited that is UNVERIFIED).
Structure (one section per stage, in this order):
1. The problem in one page (what AoA/AoD are, why a phone network needs them, why mmWave makes it hard).
2. How the receiver measures (antennas, beams, the 16x16 grid).
3. Real hardware is imperfect (per-antenna errors, position errors, phase noise, coupling) - why it matters.
4. ADCs and bits: what an ADC is, what "1-bit" means, why low-bit ADCs are used (power), what papers show (cite the verified ADC references), why NOMP struggles with them.
5. The classical champion: NOMP, explained step by step; why it is so strong on ideal hardware (credit Mamandipoor et al. 2016).
6. Our network, stage by stage: IABC / IABC-v2 corrector, CNN heatmap, decoder unfolded from NOMP, (unfolded v2 if built), the hybrid with physics-model ML. For each: what it does, analogy, what is learned, what the evidence shows.
7. How we tested fairly: the benchmark, the metric (Pd paper-style vs strict, with a worked example), confidence intervals in plain words, fresh data, the adversarial checks.
8. Results: what beats what, where, by how much (figures), including the honest negatives (NOMP is best on ideal hardware; network alone never beats the best classical method after the checks; what survived).
9. How we compare with the published papers (Lloria et al. U-Net / ResNet; others) - same benchmark, same metric, what the comparison does and does not mean.
10. What this means / the story for the journal (framings, in plain words) and limitations.
11. Glossary of every keyword; 12. References.
After compiling, render every page to PNG (PyMuPDF, into C:/Users/Milon/iabr_deep_research_cache/explainer/pages/) and look at them to fix overfull boxes, broken figures and layout problems; recompile until clean. Return the PDF path, page count and a section list.`

const CHECK = (w) => `${CTX}

EXPLAINER BUILD RESULT:
${w}

YOUR TASK (folder ${EX}/check): check the explainer PDF ${EX}/IABR-Net_Explainer.pdf and its source ${EX}/doc/explainer.tex. 1) TRACE: every number and every claim about prior work must appear in ${EX}/facts/VALIDATED_FACTS.md with the same value and meaning; list any that do not. 2) CITATIONS: every \\cite resolves, every reference is verified, nothing UNVERIFIED is presented as fact. 3) READABILITY: read it as a non-technical person - flag every unexplained term, missing example, confusing figure or paragraph, and anything that overstates the results. 4) LAYOUT: render pages (PyMuPDF) and flag layout problems. Then FIX the issues directly in explainer.tex / figures, recompile with tectonic, and re-check until clean. Write ${EX}/check/CHECK_REPORT.md (what was checked, what was fixed). Return a short summary.`

const READY = (adj, review, check) => `${CTX}

VALIDATION RESULT:
${adj}

HOSTILE REVIEW:
${review}

EXPLAINER CHECK:
${check}

YOUR TASK (folder ${EX}): write the JOURNAL-READINESS REPORT ${EX}/JOURNAL_READINESS.md and compile a short PDF version ${EX}/IABR-Net_Journal_Readiness.pdf (tectonic). Content: (1) overall verdict - READY TO SUBMIT / READY AFTER LISTED FIXES / NOT READY - with the reason in two sentences; (2) checklist with PASS/FAIL and evidence: data integrity and leakage, baseline fairness, statistics, reproducibility, literature accuracy, novelty, significance, presentation; (3) recommended framing (A/B/C) and why, with 2-3 suitable target journals and their fit (verify the journals' scope pages); (4) the exact list of remaining work before submission, in priority order, with effort estimates; (5) what must be credited and stated as limitations. Be honest: if something blocks submission, say so plainly. Return the verdict and the top-5 remaining items (<= 40 lines).`

const n = (x) => x || '(stage returned nothing - read its folder)'
phase('Facts')
const facts = await agent(FACTS, { label: 'facts sheet', phase: 'Facts', effort: 'high' })
phase('Validate')
const audits = await parallel(LENSES.map(l => () => agent(`${CTX}\n\nFACTS SHEET SUMMARY:\n${n(facts)}\n\nYOUR TASK (folder ${EX}/validation/${l.key}; write your findings to ${EX}/validation/${l.key}/FINDINGS.md with Bash): ${l.prompt}\nReturn your findings list (<= 60 lines).`, { label: l.label, phase: 'Validate', effort: 'high' })))
const auditText = LENSES.map((l, i) => `--- ${l.key} ---\n${n(audits[i])}`).join('\n')
phase('Adjudicate')
const adj = await agent(ADJ(n(facts), auditText), { label: 'adjudicate findings', phase: 'Adjudicate', effort: 'high' })
phase('Explainer')
const figs = await agent(FIGS(n(adj)), { label: 'explainer figures', phase: 'Explainer', effort: 'high' })
const w = await agent(WRITE(n(adj), n(figs)), { label: 'write + compile explainer PDF', phase: 'Explainer', effort: 'high' })
const chk = await agent(CHECK(n(w)), { label: 'trace, readability, layout check', phase: 'Explainer', effort: 'high' })
phase('Readiness')
const ready = await agent(READY(n(adj), n(audits[3]), n(chk)), { label: 'journal-readiness report', phase: 'Readiness', effort: 'high' })
return { facts, audits: auditText, adjudication: adj, figures: figs, explainer: w, check: chk, readiness: ready }
