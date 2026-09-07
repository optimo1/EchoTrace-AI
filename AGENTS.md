# Autonomous Agent Operating Guidelines (`agent.md`)

## 1. Role & Objective
You are an autonomous engineering agent executing the **EchoTrace AI** project via OpenCode. Your objective is to implement the system strictly according to `SPECIFICATION.md` (or `plan.md`) following a rigorous test-driven, secure, and incremental workflow.

---

## 2. Core Execution Protocol

You must execute work strictly step-by-step. Do not jump ahead, do not implement multiple substeps simultaneously, and never mark items complete without programmatic verification.

```text
[Select Next Substep]
         │
         ▼
[Implement Single Substep]
         │
         ▼
[Run Correctness & Security Checks]
         │
    Passed? ──No──► [Fix Errors & Re-verify]
         │
        Yes
         ▼
[Mark Substep as Completed: [x]]
         │
         ▼
[Are all Substeps in Milestone complete?]
         │
    Yes  │  No ──► (Loop to Next Substep)
         ▼
[Milestone-Level Integration Verification]
         │
    Passed? ──No──► [Resolve Integration Bugs]
         │
        Yes
         ▼
[Mark Entire Milestone as Completed: [x]]
```

---

## 3. Step-by-Step Execution Rules

### Phase A: Substep Execution
1. **Identify the Target:** Locate the first unchecked item `[ ]` in the current active milestone inside `SPECIFICATION.md` / `plan.md`.
2. **Implement Incrementally:** Write the minimal, production-grade code required to accomplish that exact substep. Do not prematurely write code for future steps.
3. **Automated Verification:**
   - **Correctness:** Write or execute unit tests or sanity scripts to prove the code functions as specified (e.g., test slice lengths, ensure proper sample rate conversion, check output tensor shapes).
   - **Security & Hygiene:**
     - Validate input extensions and sanitize filenames.
     - Ensure temporary audio buffers/files are deleted immediately after inference using `tempfile` / context managers to prevent disk exhaustion.
     - Avoid hardcoding absolute paths or secrets.
     - Check memory usage to prevent Out-Of-Memory (OOM) crashes during long audio slicing.
4. **Mark Substep Done:** Only after all checks pass cleanly, edit the plan file and change `[ ]` to `[x]` for that specific substep.

---

### Phase B: Milestone Verification & Promotion
1. **Trigger Condition:** Once every substep within the active milestone is checked `[x]`, halt feature development.
2. **Execute Milestone Validation:**
   - Run an integration test covering the entire milestone scope (e.g., end-to-end slice-to-score pipeline, full Streamlit UI rendering without runtime exceptions).
   - Verify performance benchmarks (e.g., ensure chunk processing runs efficiently on CPU).
3. **Promote Milestone:**
   - If validation passes, mark the parent milestone checkbox as completed `[x]`.
   - Provide a concise summary of the verified milestone before moving to the next one.
   - If validation fails, fix the issue immediately before starting the next milestone.

---

## 4. Coding Standards & Boundaries
- **Language & Runtime:** Python 3.11+.
- **Inference Efficiency:** Default to `Spectra0` via `jabberjay` for fast CPU evaluation.
- **Resource Caching:** All model instances must be cached via `@st.cache_resource` in Streamlit.
- **Non-Destructive Changes:** Never overwrite or delete user audio files. Use in-memory byte streams or temporary files.
- **Explainability Integrity:** Ensure start and end timestamps match the original audio track accurately so the Evidence Map timeline syncs 1:1 with audio playback.
