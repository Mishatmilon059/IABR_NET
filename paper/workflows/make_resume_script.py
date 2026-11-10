"""Generates lean_resume.js (portable: root passed as args.proj, finished stages skippable via args.skip) from lean.js.
Run:  python make_resume_script.py     (needs only lean.js next to it)"""
import os
BS = chr(92)
here = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(here, 'lean.js'), encoding='utf-8').read()
D2 = 'D:' + BS * 2 + 'ai_ml_project' + BS * 2            # D:\\ai_ml_project\\   as written inside the JS source


def rep(old, new, must=True):
    global s
    if must and old not in s:
        raise SystemExit('anchor not found: ' + old[:60])
    s = s.replace(old, new)


# 1. portable root + skip set
rep("const P = '" + D2 + "paper" + BS * 2 + "'",
    "const PROJ = (args && args.proj) || '" + D2 + "'\nconst P = PROJ + 'paper" + BS * 2 + "'\nconst skip = new Set((args && args.skip) || [])")
n = s.count(D2)
default_line = "(args && args.proj) || '" + D2 + "'"
s = s.replace(default_line, '@@KEEP@@')
s = s.replace(D2, '${PROJ}')
s = s.replace('@@KEEP@@', default_line)
# 2. skill reference path used by the final reviewer
old_gates = "${'C:" + BS * 2 + "Users" + BS * 2 + "Milon" + BS * 2 + ".claude" + BS * 2 + "skills" + BS * 2 + "ieee-paper-orchestrator" + BS * 2 + "references" + BS * 2 + "gates.md'}"
rep(old_gates, "${P}skill_backup" + BS * 2 + "references" + BS * 2 + "gates.md")
# 3. skipping of finished groups and stages
rep("async function runGroup(id, { write = true, reReview = true } = {}) {",
    "async function runGroup(id, { write = true, reReview = true } = {}) {\n  if (skip.has(id)) return { id, status: 'SKIPPED (done earlier)', open: [] }")
rep("const cc = await agent(", "const cc = skip.has('crosscheck') ? null : await agent(")
rep("await agent(`You are the FIX AGENT.", "if (!skip.has('crosscheck')) await agent(`You are the FIX AGENT.")
rep("const figP = agent(", "const figP = skip.has('figures') ? Promise.resolve() : agent(")
rep("const bibP = agent(", "const bibP = skip.has('bibliography') ? Promise.resolve() : agent(")
rep("await agent(`You are the LATEX ENGINEER", "if (!skip.has('latex')) await agent(`You are the LATEX ENGINEER")
rep("const fr = await agent(", "const fr = skip.has('finalreview') ? null : await agent(")
rep("await agent(`You are the FINAL FIX AGENT.", "if (fr) await agent(`You are the FINAL FIX AGENT.")
rep("name: 'ieee-paper-lean',", "name: 'ieee-paper-lean-resume',")
open(os.path.join(here, 'lean_resume.js'), 'w', encoding='utf-8').write(s)
print('lean_resume.js written; replaced', n, 'absolute-root occurrences')
