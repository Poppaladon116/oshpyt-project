\# RNN Tier 2 Plan



\## First component family



BADGE



\## Initial vocabulary



\- BADGE

\- COLOR\_BLUE

\- COLOR\_GREEN

\- COLOR\_RED

\- COLOR\_YELLOW

\- TEXT\_NEW

\- TEXT\_SUCCESS

\- TEXT\_ERROR

\- TEXT\_WARNING



\## Supported sequences



\- COMPONENT\_START BADGE COLOR\_BLUE TEXT\_NEW COMPONENT\_END

\- COMPONENT\_START BADGE COLOR\_GREEN TEXT\_SUCCESS COMPONENT\_END

\- COMPONENT\_START BADGE COLOR\_RED TEXT\_ERROR COMPONENT\_END

\- COMPONENT\_START BADGE COLOR\_YELLOW TEXT\_WARNING COMPONENT\_END



\## Requirements



\- BADGE must be a valid top-level component trigger.

\- Each supported sequence must have a deterministic HTML template.

\- The API fixture suite must include one case per badge variant.

\- Unknown tokens, blank prompts, and prompts with no component trigger must retain the v4.1 rejection behavior.

\- Retrained model metadata and weight artifacts must match the expanded vocabulary.



\## Quality gates



\- npm run test:api

\- npm test

\- git diff --check

