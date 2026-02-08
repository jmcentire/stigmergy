"""Finding delivery — how the mesh's output reaches the organization.

ARCHITECTURAL DECISION: Local annotation, not workspace pollution.
──────────────────────────────────────────────────────────────────

The mesh stores all findings, annotations, and decay tracking in
.stigmergy/ — a local directory that leadership consults deliberately.
Findings are NOT deposited into existing work artifacts (PR comments,
Linear ticket annotations, Slack messages, repo files).

Rationale:

1. Digital storage topology is isomorphic. Bits in annotations.json
   keyed by "linear/INT-332" carry the same information as a comment
   row in Linear's database joined to that issue. The persistence
   layer doesn't determine whether the information is "stigmergic" —
   the encounter context does.

2. Bot annotations on work artifacts develop antibodies faster than
   any other form of organizational communication. Engineers resolve
   automated PR comments without reading them after the third false
   positive. The selection pressure against bot noise is fierce.

3. The mesh produces observations about knowledge gaps, not the
   missing knowledge itself. A PR comment saying "this coupling
   exists" is not the interface contract whose absence constitutes
   the gap. Depositing observations into work artifacts creates the
   illusion of action (the comment exists!) without the substance
   (the coupling is still unmanaged).

4. The .stigmergy/ mirror is consulted in contexts where attention
   for organizational health is already allocated — the morning
   review, the weekly sync. This is Ocasio's situated attention
   working FOR the system: findings arrive when the recipient is
   already in "organizational perception" mode.

FUTURE: Artifact delivery (stubbed, not built).
────────────────────────────────────────────────

The separate concern is creating actual organizational artifacts —
Linear tickets, interface contracts, dependency diagrams — that the
team encounters naturally in the course of work. These are not
annotations on existing artifacts; they are NEW artifacts whose
existence IS the organizational knowledge that was missing.

This also solves the feedback loop: a created ticket that gets
closed is a state change the mesh can observe. A created interface
contract that gets imported is a state change. The decay tracker
can close the loop without requiring manual CLI feedback.

The delivery module stubs this interface for future implementation.
"""
