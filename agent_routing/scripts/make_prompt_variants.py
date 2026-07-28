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

GEN2LEGAL = [
    ("convert an expert-level multiple-choice question",
     "convert a legal reasoning multiple-choice question"),
    ('"<short category such as diagnosis, calculation, mechanism, definition, classification, '
     'next_step, comparison, cause_effect, application, prevention, or other>"',
     '"<short category such as issue_spotting, rule_application, statutory_interpretation, '
     'contract_analysis, classification, precedent_comparison, element_analysis, '
     'procedural_posture, or other>"'),
    ('"<compact domain knowledge slot needed to evaluate the question>"',
     '"<compact legal rule, doctrinal element, statutory provision, or definition '
     'needed to evaluate the question>"'),
]

GEN2MMLU = [
    ("convert an expert-level multiple-choice question",
     "convert an academic multiple-choice question spanning diverse subjects "
     "(STEM, humanities, social sciences, business, and law)"),
    ('"<short category such as diagnosis, calculation, mechanism, definition, classification, '
     'next_step, comparison, cause_effect, application, prevention, or other>"',
     '"<short category such as calculation, definition, mechanism, classification, comparison, '
     'cause_effect, application, interpretation, quantitative_analysis, or other>"'),
    ('"<compact domain knowledge slot needed to evaluate the question>"',
     '"<compact subject-matter fact, formula, definition, or principle '
     'needed to evaluate the question>"'),
]

_RT_OLD = ("You are the Reasoner sub-agent.\n\nGiven a question (and choices, optional context), "
           "produce a short neutral scaffold.")

RUNTIME_SCI = [(_RT_OLD,
    "You are the Reasoner sub-agent for graduate-level science (physics, chemistry, biology) "
    "multiple-choice questions.\n\nGiven a science question (and choices, optional context), "
    "produce a short neutral scaffold focusing on scientific principles, formulas, and mechanisms.")]

RUNTIME_LEGAL = [(_RT_OLD,
    "You are the Reasoner sub-agent for legal reasoning multiple-choice questions.\n\n"
    "Given a legal question (and choices, optional context), produce a short neutral scaffold "
    "focusing on the governing rule, its doctrinal elements, and how the facts map onto them.")]

RUNTIME_MMLU = [(_RT_OLD,
    "You are the Reasoner sub-agent for academic multiple-choice questions across diverse "
    "subjects (STEM, humanities, social sciences, business, and law).\n\n"
    "Given a question (and choices, optional context), produce a short neutral scaffold "
    "focusing on the relevant subject-matter principles, definitions, and computations.")]

def apply(src, dst, edits):
    text = (V / src).read_text()
    for i, (old, new) in enumerate(edits, 1):
        n = text.count(old)
        if n != 1:
            sys.exit(f"[FAIL] {src} -> {dst} edit {i}: matched {n}, expected 1\n  want: {old[:100]}")
        text = text.replace(old, new)
    (V / dst).write_text(text)
    print(f"[OK] {dst}")

apply("reasoner_medical.py", "reasoner_generic.py", MED2GEN)
apply("reasoner_generic.py", "reasoner_science.py", GEN2SCI)
apply("reasoner_generic.py", "reasoner_legal.py",   GEN2LEGAL)
apply("reasoner_generic.py", "reasoner_mmlu.py",    GEN2MMLU)

apply("runtime_prompts_neutral.py", "runtime_prompts_science.py", RUNTIME_SCI)
apply("runtime_prompts_neutral.py", "runtime_prompts_legal.py",   RUNTIME_LEGAL)
apply("runtime_prompts_neutral.py", "runtime_prompts_mmlu.py",    RUNTIME_MMLU)
