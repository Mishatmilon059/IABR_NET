export const meta = {
  name: 'ieee-paper-lean-resume',
  description: 'Lean IEEE paper pipeline: write remaining sections with one combined reviewer each, then cross-check, figures, bibliography, LaTeX and final review',
  phases: [
    { title: 'Check III+II', detail: 'one combined review of the drafted sections' },
    { title: 'IV+V', detail: 'decoders and hybrid' },
    { title: 'VI', detail: 'experimental setup' },
    { title: 'VII', detail: 'results' },
    { title: 'VIII-X', detail: 'discussion, limitations, conclusion, appendix A' },
    { title: 'I+Abstract', detail: 'written last' },
    { title: 'Cross-check', detail: 'consistency, numbers, equations, floats' },
    { title: 'Figures+Bib', detail: 'figures and verified bibliography' },
    { title: 'LaTeX', detail: 'IEEEtran build' },
    { title: 'Final review', detail: 'simulated peer review and fixes' },
  ],
}

const PROJ = (args && args.proj) || 'D:\\ai_ml_project\\'
const P = PROJ + 'paper\\'
const skip = new Set((args && args.skip) || [])
const RULES = `ABSOLUTE RULES: never fabricate data, results, numbers, equations, baselines, implementation details or citations; mark missing information [REQUIRES USER INPUT] or [NOT YET RUN]; cite audited records as [C#-###] (apply the corrections in ${P}citations\\audit_*.md) or [NV:name] for works only in NOVELTY_TABLE.md; never write first/novel/state-of-the-art. Read ${P}DECISIONS.md first and obey it (it overrides ${P}PAPER_ARCHITECTURE.md where they differ; its section 8 holds the newest evidence). TOKEN ECONOMY: the files are large; read only the parts you need (use Grep and offset/limit), never re-read a file you already read, and keep your final reply short.`
const READ = `Sources: ${P}MASTER_RESEARCH_SPEC.md (single source of truth), ${P}PAPER_ARCHITECTURE.md (plan for your section), ${P}NOVELTY_TABLE.md, ${P}evidence\\new_results\\RESULTS_NEW.md (newest results; its tables name their raw files), ${P}evidence\\audit\\ABLATION_NOMP_SELFCAL.md, ${P}citations\\topic_*_candidates.md with ${P}citations\\audit_*.md, and the code ${PROJ}IABR_v2_extracted\\IABR_v2\\notebook_src.py (decoder lines 527-647, model 420-522, metrics 315-334) and iabr2_sim.py for anything that describes an implementation (the code is the reference).`

const ISSUES = { type: 'object', properties: {
  status: { type: 'string', enum: ['SATISFIED', 'REVISE'] },
  issues: { type: 'array', items: { type: 'object', properties: {
    perspective: { type: 'string', enum: ['rigor', 'math', 'repro'] },
    severity: { type: 'string', enum: ['major', 'minor'] },
    file: { type: 'string' }, problem: { type: 'string' }, why: { type: 'string' },
    evidence: { type: 'string' }, correction: { type: 'string' }, verification: { type: 'string' } },
    required: ['perspective', 'severity', 'problem', 'why', 'evidence', 'correction', 'verification'] } } },
  required: ['status', 'issues'] }

const G = {
  III: { name: 'III. System Model and Problem Formulation', files: ['sec_III_system_model.md'], deps: [], note: 'Already drafted; you only review/revise.' },
  II: { name: 'II. Related Work', files: ['sec_II_related_work.md'], deps: [], note: 'Already drafted; you only review/revise.' },
  IV: { name: 'IV. Decoders', files: ['sec_IV_decoders.md'], deps: ['sec_III_system_model.md'],
    note: 'Describe the network-free decoder exactly as implemented (notebook_src.py 527-647): ML criterion and LS gains, per-path Newton with Hessian guard and step clip, greedy FFT initialisation (oversampling 4), cyclic re-estimation (rc = 3 by default), optional dual initialisation (needs the network of V), gated alternating-LS self-calibration with identifiable-subspace projection, (lam,tau) = (1.0,5.0), angle recovery with its end-fire fold defect, complexity. Include Algorithm 1 pseudocode, a decoder hyper-parameter table and a FLOAT for the decoder flowchart. State that 3 cyclic rounds are under-converged for close paths and that selfcal() with lam = 0 still runs extra cyclic rounds (forward reference to VII-E). Call it an in-house adaptation of NOMP and credit it.' },
  V: { name: 'V. Learned Initialiser and Corrector (IABR-v2) as the Compared Hybrid', files: ['sec_V_hybrid.md'], deps: ['sec_III_system_model.md'],
    note: 'About 1 page: trunk, IABC, heatmap head, training objective and recipe, how the decoder consumes the output, what is and is not retrained in the ablations. Cap statements about IABC per N4 of the architecture. Parameter table with the verified counts (191,702 total, 458 in IABC).' },
  VI: { name: 'VI. Experimental Setup', files: ['sec_VI_experimental_setup.md'], deps: ['sec_III_system_model.md', 'sec_IV_decoders.md', 'sec_V_hybrid.md'],
    note: 'Training stream, simulator, test banks (with seeds, sizes, SNR grids, known defects), methods compared, formal metric definitions (paper-style Pd, strict Pd, pd_source, conditional RMSE; formalise and check against TVT_Blob_Inference.py lines ~123-159 and metrics() in notebook_src.py), the paired-bootstrap procedure exactly as RESULTS_NEW.md describes it, computing environments and provenance (which machine produced which result), leakage/design-exposure statement. Do not copy the slide statement that L is always 3.' },
  VII: { name: 'VII. Results', files: ['sec_VII_results.md'], deps: ['sec_III_system_model.md', 'sec_IV_decoders.md', 'sec_V_hybrid.md', 'sec_VI_experimental_setup.md'],
    note: 'Subsections A main benchmark, B decomposition of the hybrid, C comparison with the JoS Table 2 conditions, D robustness, E close paths (use the E4 conclusion: the gain is extra cyclic refinement, not calibration), F why accuracy saturates (use the NEW joint-Fisher diagnostics, not the old diag2/diag3), G cost (measured: laptop CPU latency of the decoder; hybrid numbers from RESULTS.md on a different machine, named). Every table cell has a paired interval where RESULTS_NEW.md provides one. Strict Pd primary, paper-style Pd beside it. Write the table contents fully (real numbers from the files), FLOAT blocks for figures 5-9 with captions.' },
  VIIIX: { name: 'VIII. Discussion, IX. Limitations, X. Conclusion and Appendix A (identifiability)', files: ['sec_VIII_discussion.md', 'sec_IX_limitations.md', 'sec_X_conclusion.md', 'appendix_A_identifiability.md'], deps: ['sec_VII_results.md'],
    note: 'Discussion without new numbers; Limitations as one fact per bullet (spec section 12 updated with RESULTS_NEW.md); Conclusion restating the contributions in the same qualified words; one future-work bullet allowed, nothing from the excluded learned decoder. Appendix A: derivation that the mean phase, the linear phase ramp per side and the mean log-gain per side are not identifiable (read IABR_v2_DATASET_AUDIT\\FINAL_DATASET_AUDIT_REPORT.md section 3 and paper\\evidence\\audit\\iabc_target_variance.py); ramp = angle shift is stated as our inference; cite Ramamohan (C3-004) with its audit caveat.' },
  IABS: { name: 'I. Introduction and Abstract', files: ['sec_I_introduction.md', 'abstract.md'], deps: ['sec_II_related_work.md', 'sec_III_system_model.md', 'sec_IV_decoders.md', 'sec_VII_results.md', 'sec_VIII_discussion.md', 'sec_IX_limitations.md', 'sec_X_conclusion.md'],
    note: 'Written last. The contribution list must match section VII exactly and respect NOVELTY_TABLE.md (to the best of our knowledge only for P1/P4, once). Abstract at most 180 words, no citations, no equations, states the finding with qualifiers and scope (simulation, L given, 16x16, one network seed). Also write index terms.' },
}

const writerPrompt = (id) => { const g = G[id]; return `You are the WRITER of "${g.name}" of an IEEE journal paper. ${RULES}
${READ}
Follow the plan for your section(s) in ${P}PAPER_ARCHITECTURE.md as modified by DECISIONS.md and NOVELTY_TABLE.md. ${g.deps.length ? 'Earlier sections to stay consistent with (notation, names, numbers): ' + g.deps.map(d => P + 'sections\\' + d).join(', ') + '.' : ''}
Specific instructions: ${g.note}
Write to: ${g.files.map(f => P + 'sections\\' + f).join(', ')}. Markdown with LaTeX math, format rules in DECISIONS.md section 6, a source comment (<!-- src: ... -->) after every sentence or table row that carries a number, FLOAT blocks (caption, data source, status) at the end for figures and tables. For each design choice answer: what, why, why not the alternative, assumption, what if it fails, limitation. As long as an IEEE section of this kind needs; do not pad. Final reply: files written, word counts, FLOAT list, markers left, contradictions found (under 120 words).` }

const reviewPrompt = (id, round, prev) => { const g = G[id]; return `You are the COMBINED REVIEWER (round ${round}) for "${g.name}" of an IEEE journal paper. You did not write it and must not rewrite it. ${RULES}
${READ}
Review the file(s) ${g.files.map(f => P + 'sections\\' + f).join(', ')} from three perspectives and tag each issue with its perspective: (rigor) scientific rigor: is each choice justified, assumptions realistic, argument causal, claims supported by the evidence files, DECISIONS.md and NOVELTY_TABLE.md respected, anything overclaimed; (math) every equation, dimension, notation, sign convention, objective/loss, complexity statement and the match with the code; (repro) numbers traceable to their evidence, baselines and metrics fair, provenance and seeds stated, reproducible from the paper alone. ${g.deps.length ? 'Check consistency with ' + g.deps.map(d => P + 'sections\\' + d).join(', ') + '.' : ''}
${round > 1 ? `RE-REVIEW: re-evaluate the revised text yourself; an earlier issue counts as resolved only if the text now resolves it (the revision log is ${P}reviews\\${id}_tracker.md). Earlier issues: ${JSON.stringify(prev || [])}.` : ''}
Issues must be specific (file, quote or line, correction, how to verify). Return SATISFIED only if no major issue remains. Do not pad with praise.` }

const revisePrompt = (id, round, issues) => { const g = G[id]; return `You are the REVISION AGENT (round ${round}) for "${g.name}". ${RULES}
${READ}
Reviewer issues: ${JSON.stringify(issues)}
Revise ${g.files.map(f => P + 'sections\\' + f).join(', ')} in place so every issue is resolved or explicitly kept open with a reason; fix numbers only from evidence files; if data does not exist, scope the claim or write [NOT YET RUN]. Append to ${P}reviews\\${id}_tracker.md one line per issue: "ISSUE-${id}-r${round}-nn | perspective | severity | problem | action | RESOLVED or OPEN (reason)". Final reply: counts of resolved and open (under 60 words).` }

async function runGroup(id, { write = true, reReview = true } = {}) {
  if (skip.has(id)) return { id, status: 'SKIPPED (done earlier)', open: [] }
  const ph = id === 'III' || id === 'II' ? 'Check III+II' : id === 'IV' || id === 'V' ? 'IV+V' : id === 'VIIIX' ? 'VIII-X' : id === 'IABS' ? 'I+Abstract' : id
  if (write) await agent(writerPrompt(id), { label: `write ${id}`, phase: ph })
  const r1 = await agent(reviewPrompt(id, 1), { label: `review ${id} r1`, phase: ph, schema: ISSUES, effort: 'medium' })
  if (!r1) return { id, status: 'REVIEW-FAILED', open: [] }
  const majors = r1.issues.filter(i => i.severity === 'major').length
  if (r1.status === 'SATISFIED' && majors === 0) return { id, status: 'SATISFIED', open: r1.issues.map(i => `[${i.severity}] ${i.problem}`) }
  await agent(revisePrompt(id, 1, r1.issues), { label: `revise ${id}`, phase: ph })
  if (!reReview) return { id, status: 'REVISED-NO-RE-REVIEW', open: r1.issues.filter(i => i.severity === 'major').map(i => `[addressed? ] ${i.problem}`) }
  const r2 = await agent(reviewPrompt(id, 2, r1.issues), { label: `re-review ${id}`, phase: ph, schema: ISSUES, effort: 'medium' })
  if (!r2) return { id, status: 'RE-REVIEW-FAILED', open: [] }
  return { id, status: r2.status, open: r2.issues.map(i => `[${i.severity}] ${i.problem}`) }
}

phase('Check III+II')
const a = await parallel([() => runGroup('III', { write: false, reReview: false }), () => runGroup('II', { write: false, reReview: false })])
phase('IV+V')
const b = await parallel([() => runGroup('IV'), () => runGroup('V')])
phase('VI'); const c = await runGroup('VI')
phase('VII'); const d = await runGroup('VII')
phase('VIII-X'); const e = await runGroup('VIIIX')
phase('I+Abstract'); const f = await runGroup('IABS')
const sectionStatus = [...a, ...b, c, d, e, f].filter(Boolean)
log('Section status: ' + sectionStatus.map(s => `${s.id}:${s.status}`).join(', '))

phase('Cross-check')
const CC = { type: 'object', properties: { issues: { type: 'array', items: { type: 'object', properties: {
  severity: { type: 'string', enum: ['major', 'minor'] }, file: { type: 'string' }, location: { type: 'string' }, problem: { type: 'string' }, fix: { type: 'string' } },
  required: ['severity', 'file', 'problem', 'fix'] } } }, required: ['issues'] }
const cc = skip.has('crosscheck') ? null : await agent(`You are the CROSS-SECTION AND AUDIT AGENT of an IEEE paper pipeline. ${RULES}
Read ALL files in ${P}sections\\ (every section, appendix and abstract) and ${P}DECISIONS.md, ${P}NOVELTY_TABLE.md, ${P}evidence\\new_results\\RESULTS_NEW.md. Check and fix nothing yourself; report. (1) terminology and symbols consistent across sections (network-free decoder, NOMP+selfcal, hybrid (IABR-v2), IABC naming per DECISIONS.md); (2) equations: numbering/labels unique, every symbol defined, dimensions, sign conventions u = pi cos(psi), v = pi cos(phi), the decoder text versus notebook_src.py 527-647; (3) numbers: verify at least 40 numbers spread over all sections against the evidence files named in their src comments (list mismatches and any number without a source); (4) floats: every Fig./Table referenced in the text and defined once, numbering sequential, no orphan; (5) the contributions listed in the Introduction versus what VII, VIII and X deliver and what the evidence supports; (6) forbidden claims (N1-N14 in PAPER_ARCHITECTURE.md, first/novel/state-of-the-art, outperforms, edge/real-time/robust unqualified, anything from the excluded learned decoder); (7) citation tags: every [C#-###] exists and carries the audit corrections, [NV:...] tags listed; (8) the abstract agrees with the results. Write the full report to ${P}CROSS_CHECK.md and return the issues.`, { label: 'cross-check', phase: 'Cross-check', schema: CC, effort: 'medium' })
const ccIssues = cc ? cc.issues : []
if (!skip.has('crosscheck')) await agent(`You are the FIX AGENT. ${RULES}
Apply these cross-check issues to the files in ${P}sections\\ (edit in place; fix numbers only from the evidence files; if a statement cannot be supported, scope it or remove it): ${JSON.stringify(ccIssues)}
Append one line per issue to ${P}reviews\\crosscheck_tracker.md (issue | action | RESOLVED or OPEN with reason). Final reply under 60 words.`, { label: 'fix cross-check', phase: 'Cross-check' })

phase('Figures+Bib')
const figP = skip.has('figures') ? Promise.resolve() : agent(`You are the FIGURE AGENT of an IEEE paper pipeline. ${RULES}
Read the FLOAT blocks at the end of every file in ${P}sections\\ to see which figures the text expects. Create them in ${P}latex\\figs\\ as vector PDF plus 300-dpi PNG with matplotlib, readable at IEEE single-column (3.5 in) or double-column width, colour-blind-safe, labelled axes with units, no title inside the figure, legends not covering data. Data ONLY from ${P}evidence\\new_results\\ (bank_json, preds, e4, e5 files, RESULTS_NEW.md) and ${PROJ}IABR_v2_extracted\\IABR_v2\\outputs\\ (cache, results json); never type numbers by hand if a file holds them. Needed: Pd and RMSE versus SNR on the official bank (U-Net, hybrid, network-free, NOMP+selfcal) with the interval where available; ablation bars with intervals; robustness panels (gain, phase, L, separation, SNR range); failure taxonomy and observed versus Cramer-Rao-limit prediction per SNR; end-fire error histogram showing fold errors; decoder flowchart (greedy FFT init, Newton, cyclic, gated self-calibration, optional heatmap seeds with minimum-residual selection, arccos); observation-chain schematic; for the hybrid architecture copy ${PROJ}presentation_dataset_generation\\slides\\S19_iabr_net_code_faithful.png after checking it against notebook_src.py lines 420-522 (note any discrepancy); for the data-flow figure draw a small corrected version (L varies in training, 1 to 9, fixed in the test banks). View every figure after drawing it and fix overlaps. Write ${P}latex\\figs\\FIGURES.md listing each figure number, filename, caption text, and the exact data files used. Final reply under 100 words.`, { label: 'figures', phase: 'Figures+Bib' })
const bibP = skip.has('bibliography') ? Promise.resolve() : agent(`You are the BIBLIOGRAPHY AGENT of an IEEE paper pipeline. ${RULES}
Collect every citation tag used in ${P}sections\\ ([C#-###] and [NV:name]). Build ${P}latex\\references.bib (BibTeX, IEEEtran-compatible) and ${P}latex\\cite_map.json (tag -> bibkey). For [C#-###] use the records in ${P}citations\\topic_*_candidates.md with the corrections of ${P}citations\\audit_*.md (authors, DOI, pages, years) and merge duplicates (list in PAPER_ARCHITECTURE.md section 3 II). For every [NV:name] tag verify the work through Crossref (https://api.crossref.org/works/<DOI> or a title query) or another primary source with WebFetch/WebSearch (load via ToolSearch 'select:WebSearch,WebFetch'); take metadata only from what you actually saw; never invent a field; omit a field you cannot confirm. Entries you cannot verify are NOT put in the .bib: instead write ${P}latex\\unverified_citations.md listing the tag, where it is cited, and the sentence, so the text can be scoped. Write ${P}latex\\citation_report.md (tag, bibkey, status VERIFIED/PASS-WITH-LIMITS/UNVERIFIED, source). Final reply under 100 words with the counts.`, { label: 'bibliography', phase: 'Figures+Bib' })
await Promise.all([figP, bibP])

phase('LaTeX')
if (!skip.has('latex')) await agent(`You are the LATEX ENGINEER of an IEEE paper pipeline. ${RULES}
Convert the paper to IEEEtran journal LaTeX: ${P}latex\\main.tex (documentclass[journal]{IEEEtran}). Use the abstract and index terms from ${P}sections\\abstract.md, sections I to X and Appendix A in order. Strip the <!-- src: --> comments. Convert tags to \\cite{key} with ${P}latex\\cite_map.json (tags without an entry become plain text "[reference to be added]" and are listed in ${P}latex\\unverified_citations.md). Real LaTeX equations with labels, tables from the markdown tables (IEEE conventions, booktabs, units, \\label), figures from ${P}latex\\figs\\ with captions from FIGURES.md/FLOAT blocks and \\label, every float referenced with \\ref in the text, Algorithm 1 as an algorithm/algorithmic block. Authors, affiliations, funding, IABC expansion, code/data availability: keep the [REQUIRES USER INPUT] placeholders visibly (e.g. \\author{[REQUIRES USER INPUT]}). Do not change any claim or number; if the Markdown has an error, note it in ${P}latex\\build_notes.md instead. Check for a TeX installation (where pdflatex / latexmk; also C:\\Program Files\\MiKTeX* or TeX Live); if available compile (pdflatex, bibtex, pdflatex x2), fix errors and warnings (undefined references, overfull boxes that matter) and keep main.pdf; if not available, say so in build_notes.md and verify by static checks (balanced environments, labels defined once, all \\ref and \\cite targets exist). Final reply: compile status, page count if compiled, warnings left (under 100 words).`, { label: 'latex', phase: 'LaTeX' })

phase('Final review')
const FR = { type: 'object', properties: { issues: { type: 'array', items: { type: 'object', properties: {
  severity: { type: 'string', enum: ['major', 'minor'] }, category: { type: 'string' }, file: { type: 'string' }, problem: { type: 'string' }, fix: { type: 'string' } },
  required: ['severity', 'category', 'problem', 'fix'] } } }, required: ['issues'] }
const fr = skip.has('finalreview') ? null : await agent(`You are the FINAL PEER REVIEWER and PAPER-INTEGRITY AGENT. ${RULES}
Read the complete manuscript (${P}latex\\main.tex, or the section files if it does not compile), ${P}DECISIONS.md, ${P}NOVELTY_TABLE.md, ${P}CROSS_CHECK.md, ${P}latex\\citation_report.md, ${P}latex\\unverified_citations.md, ${P}latex\\build_notes.md. Act as a hostile IEEE reviewer: list Major and Minor comments, reproducibility, mathematical, experimental, novelty and citation concerns. Then answer the 15 integrity questions of the orchestrator skill (${P}skill_backup\\references\\gates.md) with evidence, say what a hostile reviewer would attack first, and fill the quality-gate checklist honestly (a box stays unchecked if it is not met; known unmet items: publisher pages not read, GPU experiments, user inputs). Write ${P}FINAL_REVIEW.md. Return the issues that can still be fixed without new experiments or user input.`, { label: 'final review', phase: 'Final review', schema: FR })
const frIssues = fr ? fr.issues : []
if (fr) await agent(`You are the FINAL FIX AGENT. ${RULES}
Apply these final-review issues to the section files in ${P}sections\\ AND to ${P}latex\\main.tex consistently (edit in place; no new numbers; scope or remove unsupported claims): ${JSON.stringify(frIssues.filter(i => i.severity === 'major' || true))}
Then recompile if a TeX installation exists and fix errors. Append a resolution log to ${P}FINAL_REVIEW.md (issue | action | RESOLVED/OPEN) and list the items that remain blocked on the user. Final reply under 120 words.`, { label: 'final fix', phase: 'Final review' })

return { sections: sectionStatus, crossCheckIssues: ccIssues.length, finalReviewIssues: frIssues.length }
