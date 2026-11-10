"""Shows what is already written and prints the `args` to pass to lean_resume.js so that finished work is skipped.
Run from anywhere:  python status.py
A section group counts as DONE when all its files exist and are non-trivial. A stage (crosscheck, figures,
bibliography, latex, finalreview) counts as done when its output file exists. If you doubt a section, delete it from
the skip list: its writer and reviewer will then run again."""
import os, json, glob
paper = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
proj = os.path.abspath(os.path.join(paper, '..')) + os.sep
SEC = os.path.join(paper, 'sections')
GROUPS = {'III': ['sec_III_system_model.md'], 'II': ['sec_II_related_work.md'], 'IV': ['sec_IV_decoders.md'], 'V': ['sec_V_hybrid.md'],
          'VI': ['sec_VI_experimental_setup.md'], 'VII': ['sec_VII_results.md'],
          'VIIIX': ['sec_VIII_discussion.md', 'sec_IX_limitations.md', 'sec_X_conclusion.md', 'appendix_A_identifiability.md'],
          'IABS': ['sec_I_introduction.md', 'abstract.md']}
STAGES = {'crosscheck': 'CROSS_CHECK.md', 'figures': os.path.join('latex', 'figs', 'FIGURES.md'), 'bibliography': os.path.join('latex', 'references.bib'),
          'latex': os.path.join('latex', 'main.tex'), 'finalreview': 'FINAL_REVIEW.md'}
skip = []
print('SECTIONS')
for g, files in GROUPS.items():
    st = []
    for f in files:
        p = os.path.join(SEC, f)
        st.append(os.path.getsize(p) if os.path.exists(p) else 0)
    done = all(x > 1500 for x in st)
    tr = os.path.join(paper, 'reviews', g + '_tracker.md')
    print(f'  {g:6s} {"DONE " if done else "TODO "} sizes={st} tracker={"yes" if os.path.exists(tr) else "no"}')
    if done: skip.append(g)
print('STAGES')
for s, f in STAGES.items():
    p = os.path.join(paper, f)
    ok = os.path.exists(p) and os.path.getsize(p) > 200
    print(f'  {s:12s} {"DONE " if ok else "TODO "} {f}')
    if ok: skip.append(s)
args = {'proj': proj, 'skip': skip}
print('\nPass this to the Workflow tool (scriptPath = ' + os.path.join(paper, 'workflows', 'lean_resume.js') + '):')
print(json.dumps(args))
print('\nNote: sections written but never reviewed are listed as DONE; re-review them by removing them from "skip".')
