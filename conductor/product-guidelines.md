# Product Guidelines: registration-toolkit

## Voice and Tone

Concise and direct. Documentation, CLI help, and log messages say what is
needed and stop. In-code comments are need-to-know (blocks of 2-3 lines max);
docstrings are written for humans and are descriptive, not exhaustive.

## Design Principles

1. **Correctness over speed.** A subtly wrong output (mirrored, misaligned,
   wrong UV origin) is worse than a slow one. State invariants and check them.
2. **One canonical type per concept.** `rt::Mesh`, `rt::UVMap`, top-left UV
   origin. Convert to other representations (VTK, ITK, file formats) only at I/O
   and algorithm boundaries.
3. **Explicit opt-in.** Defaults stay safe and backward compatible. Behavior that
   asserts a precondition about the input (e.g. canonical orientation) is
   opt-in.
4. **Record decisions.** When domain terms or architecture change, update
   `CONTEXT.md` (glossary) and add or amend an ADR in `docs/adr/`. Use
   `/mattpocock-skills:grill-with-docs` for this. It is user-invoked only;
   agents reach the same workflow by invoking `mattpocock-skills:grilling` and
   `mattpocock-skills:domain-modeling`.
