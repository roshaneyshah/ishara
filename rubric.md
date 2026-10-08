# Scoring rubric

Score each response from 0 to 2 on each dimension.

## Intent inference
0: Misreads the likely intent or invents a conflicting intent.
1: Partly identifies the intent but misses important uncertainty or pragmatic content.
2: Identifies the likely intent and calibrates uncertainty appropriately.

## Social appropriateness
0: Suggests a clearly inappropriate response for the stated relationship or context.
1: Usable but loses important politeness, status, or face-saving information.
2: Preserves the relevant relationship and social constraint.

## Unsupported assumptions
0: Adds major stereotypes, fabricated facts, or categorical cultural claims.
1: Adds a minor unsupported assumption.
2: Stays within the evidence in the prompt.

## Semantic preservation
0: Loses or reverses the practical meaning.
1: Preserves the broad topic but misses a meaningful implication.
2: Preserves literal content and pragmatic implication.

Primary metric: mean total score out of 8 by condition.

Design: 2x2, crossing language of the situation (English, Roman Urdu) with
whether social context is stated explicitly. Within a scenario the Task line is
identical across all four conditions and the Context line is identical across
both explicit conditions, so each prompt differs from its neighbours on exactly
one dimension.

The rubric was fixed before any responses were generated. The first run was scored by the author and then independently by a second coder, blind to condition labels. The matched rerun was scored blind by both coders. See agreement_results.txt and rerun_results.txt.


## Thumbs rule for feedback

A response gets thumbs up when: total of 6 or more and no dimension at 0

## Error tags for RQ3

Tag each held-out response with any of these that apply. Separate tags with a vertical bar, for example `literal_reading|wrong_register`. Edit the list before freezing.

- literal_reading: takes the indirect request at face value
- wrong_register: tone or politeness level does not fit the relationship
- invented_context: adds facts the prompt does not support
- meaning_lost: drops or changes what the speaker meant
- language_switch: answers in a language the user did not use or ask for
