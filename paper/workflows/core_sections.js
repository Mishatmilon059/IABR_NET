export const meta = {
  name: 'ieee-paper-core-sections',
  description: 'Write sections III, II, IV, V, VI of the IEEE paper; each section goes writer -> 3 hostile reviewers -> revision -> re-review (max 2 revisions)',
  phases: [
    { title: 'III + II', detail: 'system model and related work in parallel' },
    { title: 'IV', detail: 'decoders' },
    { title: 'V', detail: 'hybrid IABR-v2' },
    { title: 'VI', detail: 'experimental setup' },
  ],
}

const P = 'D:\\ai_ml_project\\paper\\'
const RULES = `ABSOLUTE RULES: never fabricate data, results, numbers, equations, baselines, implementation details or citations; mark missing information [REQUIRES USER INPUT] or [NOT YET RUN]; cite only audited records as [C#-###] (apply the corrections in ${P}citations\\audit_*.md) or the tag [NV:name] for works only in NOVELTY_TABLE.md; never write first/novel/state-of-the-art. Read ${P}DECISIONS.md first and obey it (it overrides ${P}PAPER_ARCHITECTURE.md where they differ).`
const READ = `Inputs: ${P}MASTER_RESEARCH_SPEC.md (single source of truth), ${P}PAPER_ARCHITECTURE.md (the plan for your section), ${P}NOVELTY_TABLE.md, ${P}citations\\topic_*_candidates.md and ${P}citations\\audit_*.md, ${P}evidence\\README.md, ${P}evidence\\audit\\ABLATION_NOMP_SELFCAL.md, and the source code D:\\ai_ml_project\\IABR_v2_extracted\\IABR_v2\\notebook_src.py / iabr2_sim.py where the text describes an implementation. Read only what your section needs.`

const REVIEWERS = [
  { key: 'rigor', name: 'SCIENTIFIC RIGOR REVIEWER', brief: 'Act like an extremely demanding IEEE journal reviewer. Ask: why was this method/choice selected, is it theoretically justified, which assumptions are made and are they realistic, is the argument causal and complete rather than descriptive, are conclusions supported by the evidence, is the alternative considered, is anything overclaimed relative to the evidence files and to DECISIONS.md and NOVELTY_TABLE.md.' },
  { key: 'math', name: 'MATHEMATICAL / TECHNICAL REVIEWER', brief: 'Attack every technical detail: equations, dimensions, notation consistency, tensor shapes, derivations, objectives and losses, complexity statements, parameter definitions, boundary conditions, sign conventions, physical interpretation. For every equation ask why this form, where it comes from, what each term is, which assumption it needs, what happens if it fails. Verify the text against the code (notebook_src.py / iabr2_sim.py) wherever it describes an implementation; the code is the reference.' },
  { key: 'repro', name: 'EXPERIMENTAL / REPRODUCIBILITY REVIEWER', brief: 'Attack dataset construction, preprocessing, splits and leakage, hyper-parameters, baselines and comparison fairness, metrics, statistics, seeds, hardware/software provenance, ablation logic. Verify every number in the text against its cited evidence file and flag any number without a source. Ask: could another researcher reproduce this from the paper alone? List exactly what is missing.' },
]

const ISSUES = { type: 'object', properties: {
  status: { type: 'string', enum: ['SATISFIED', 'REVISE'] },
  summary: { type: 'string' },
  issues: { type: 'array', items: { type: 'object', properties: {
    severity: { type: 'string', enum: ['major', 'minor'] },
    problem: { type: 'string' }, why: { type: 'string' }, evidence: { type: 'string' },
    correction: { type: 'string' }, verification: { type: 'string' } },
    required: ['severity', 'problem', 'why', 'evidence', 'correction', 'verification'] } } },
  required: ['status', 'issues'] }

const SECTIONS = {
  III: { file: 'sec_III_system_model.md', name: 'III. System Model and Problem Formulation', deps: [] },
  II: { file: 'sec_II_related_work.md', name: 'II. Related Work', deps: [] },
  IV: { file: 'sec_IV_decoders.md', name: 'IV. Decoders', deps: ['sec_III_system_model.md'] },
  V: { file: 'sec_V_hybrid.md', name: 'V. Learned Initialiser and Corrector (IABR-v2) as the Compared Hybrid', deps: ['sec_III_system_model.md', 'sec_IV_decoders.md'] },
  VI: { file: 'sec_VI_experimental_setup.md', name: 'VI. Experimental Setup', deps: ['sec_III_system_model.md', 'sec_IV_decoders.md', 'sec_V_hybrid.md'] },
}

const writerPrompt = (id) => {
  const s = SECTIONS[id]
  return `You are AGENT_SECTION_${id}, the WRITER of section "${s.name}" of an IEEE journal paper. ${RULES}
${READ}
Find your section in ${P}PAPER_ARCHITECTURE.md (heading "${s.name.split('.')[0]}.") and follow its purpose, outline, citations (with the audit corrections), owned equations/figures/tables and dependencies, as modified by DECISIONS.md and NOVELTY_TABLE.md. ${s.deps.length ? 'Earlier sections you must stay consistent with (notation, names, numbers): ' + s.deps.map(d => P + 'sections\\' + d).join(', ') + '.' : 'You are written first or in parallel: define the notation cleanly, later sections will adopt it.'}
Write the section to ${P}sections\\${s.file} (Markdown, LaTeX math, format rules in DECISIONS.md section 6; a source comment after every sentence or table row that carries a number; FLOAT blocks at the end for figures/tables with caption, data source and status). Answer for every design choice: what, why, why not the alternative, which assumption, what if it fails, the limitation. Length: as long as an IEEE section of this kind needs (typically 600-1400 words plus equations and tables); do not pad.
Return JSON-like summary as your final text: file written, word count, list of FLOAT blocks, list of [REQUIRES USER INPUT]/[NOT YET RUN]/[CITATION NEEDED] markers you left, and anything in the evidence that contradicts the architecture.`
}

const reviewPrompt = (id, rv, round, prev) => {
  const s = SECTIONS[id]
  return `You are ${rv.name} for section "${s.name}" (review round ${round}) of an IEEE journal paper. You did not write it and must not rewrite it; you only review. ${RULES}
${READ}
${rv.brief}
Read the section at ${P}sections\\${s.file}. ${s.deps.length ? 'Check consistency with ' + s.deps.map(d => P + 'sections\\' + d).join(', ') + '.' : ''}
${round > 1 ? `This is a RE-REVIEW. The revision agent claims to have addressed earlier issues (see ${P}reviews\\${id}_tracker.md). Re-evaluate the revised text yourself: an issue counts as resolved only if the text now resolves it. Do not approve merely because the text was rewritten; list any issue that remains or was introduced. Earlier issues you had raised: ${JSON.stringify(prev || [])}.` : ''}
For each issue give: severity (major = a hostile IEEE reviewer would reject or the claim is unsupported/incorrect; minor = clarity or polish), problem, why it matters, evidence (file/line/quote), required correction, verification criteria. Return status SATISFIED only if no major issue remains and you would accept this section for submission with at most minor polish; otherwise REVISE. Be specific and concise; do not pad with praise.`
}

const revisePrompt = (id, round, all) => {
  const s = SECTIONS[id]
  return `You are the REVISION AGENT for section "${s.name}" (round ${round}). ${RULES}
${READ}
Reviewer issues (JSON): ${JSON.stringify(all)}
Revise ${P}sections\\${s.file} in place so that EVERY issue is resolved or explicitly kept open with a reason. Do not delete criticism; do not weaken a claim's evidence; fix numbers only from evidence files; if an issue needs data that does not exist, replace the claim with [NOT YET RUN] or scope it honestly. Append to ${P}reviews\\${id}_tracker.md one entry per issue in this format: "ISSUE-${id}-r${round}-nn | reviewer | severity | problem | action taken | Status: RESOLVED or OPEN (reason) | verification criterion". Return a short list of resolved and open issues.`
}

async function sectionLoop(id) {
  const ph = id === 'III' || id === 'II' ? 'III + II' : id
  await agent(writerPrompt(id), { label: `write ${id}`, phase: ph })
  let reviews = []
  let prevByKey = {}
  const history = []
  for (let round = 1; round <= 3; round++) {
    const results = await parallel(REVIEWERS.map(rv => () => agent(reviewPrompt(id, rv, round, prevByKey[rv.key]), { label: `review ${id} r${round} ${rv.key}`, phase: ph, schema: ISSUES })))
    reviews = results.map((r, i) => ({ key: REVIEWERS[i].key, ...(r || { status: 'REVISE', issues: [{ severity: 'major', problem: 'reviewer failed to return', why: 'agent error', evidence: '-', correction: 'rerun review', verification: '-' }] }) }))
    history.push({ round, reviews })
    reviews.forEach(r => { prevByKey[r.key] = r.issues })
    const allSat = reviews.every(r => r.status === 'SATISFIED')
    if (allSat) { log(`${id}: all three reviewers SATISFIED in round ${round}`); break }
    const nOpen = reviews.reduce((a, r) => a + r.issues.filter(i => i.severity === 'major').length, 0)
    if (round === 3) { log(`${id}: round cap reached with ${nOpen} major issues still raised; recorded as OPEN`); break }
    log(`${id} round ${round}: ${nOpen} major issues; revising`)
    await agent(revisePrompt(id, round, reviews), { label: `revise ${id} r${round}`, phase: ph })
  }
  const last = history[history.length - 1].reviews
  return { id, rounds: history.length, final: last.map(r => ({ key: r.key, status: r.status, open: r.issues.map(i => `[${i.severity}] ${i.problem}`) })) }
}

phase('III + II')
const first = await parallel([() => sectionLoop('III'), () => sectionLoop('II')])
phase('IV'); const iv = await sectionLoop('IV')
phase('V'); const v = await sectionLoop('V')
phase('VI'); const vi = await sectionLoop('VI')
return [...first, iv, v, vi]
