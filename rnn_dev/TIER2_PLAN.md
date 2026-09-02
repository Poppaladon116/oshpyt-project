\# RNN Tier 2 Plan



\## Goal

Expand the supported component vocabulary and deterministic renderer while preserving the v4.1 API contract.



\## First component family

Choose one:

\- CARD

\- BADGE

\- INPUT

\- NAVBAR

\- MODAL



\## Required changes

\- Add tokens to training/component data

\- Add valid component sequences

\- Retrain or regenerate model artifacts

\- Update model metadata and vocabulary

\- Add renderTokens templates

\- Add API fixtures and expected HTML assertions

\- Preserve input rejection behavior



\## Quality gates

\- npm run test:api

\- npm test

\- git diff --check

