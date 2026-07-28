import sys, pathlib
V = pathlib.Path("src/subagents/prompts/variants")

MED2GEN = [
    ("convert a medical multiple-choice question",
     "convert an expert-level multiple-choice question"),
    ('"<short category such as diagnosis, therapy_selection, risk_factor, mechanism, '
     'next_step, prevention, adverse_effect, prognosis, or other>"',
     '"<short category such as diagnosis, calculation, mechanism, definition, classification, '
     'next_step, comparison, cause_effect, application, prevention, or other>"'),
    ('"<compact medical knowledge slot needed to evaluate the case>"',
     '"<compact domain knowledge slot needed to evaluate the question>"'),
]

GEN2SCI = [
    ("convert an expert-level multiple-choice question",
     "convert a graduate-level science multiple-choice question in physics, chemistry, or biology"),
    ('"<short category such as diagnosis, calculation, mechanism, definition, classification, '
     'next_step, comparison, cause_effect, application, prevention, or other>"',
     '"<short category such as calculation, mechanism, quantitative_analysis, conceptual_reasoning, '
     'formula_application, experimental_design, comparison, cause_effect, or other>"'),
    ('"<compact domain knowledge slot needed to evaluate the question>"',
     '"<compact scientific principle, formula, or domain fact (physics/chemistry/biology) '
     'needed to evaluate the question>"'),
]

RUNTIME_SCI = [
    ("You are the Reasoner sub-agent.\n\nGiven a question (and choices, optional context), "
     "produce a short neutral scaffold.",
     "You are the Reasoner sub-agent for graduate-level science (physics, chemistry, biology) "
     "multiple-choice questions.\n\nGiven a science question (and choices, optional context), "
     "produce a short neutral scaffold focusing on scientific principles, formulas, and mechanisms."),
]

def apply(src, dst, edits):
    text = (V / src).read_text()
    for i, (old, new) in enumerate(edits, 1):
        n = text.count(old)
        if n != 1:
            sys.exit(f"[FAIL] {src} -> {dst} edit {i}: matched {n}, expected 1\n  want: {old[:100]}")
        text = text.replace(old, new)
    (V / dst).write_text(text)
    print(f"[OK] {dst}")

apply("reasoner_medical.py",        "reasoner_generic.py",        MED2GEN)
apply("reasoner_generic.py",        "reasoner_science.py",        GEN2SCI)
apply("runtime_prompts_neutral.py", "runtime_prompts_science.py", RUNTIME_SCI)
