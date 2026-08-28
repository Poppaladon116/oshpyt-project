\# OSHPYT RNN v4.1 Hardened Checkpoint



Date: 2026-08-23



\## Test status

\- API regression suite: 17/17 passed

\- Valid component plans: 14

\- Input-rejection tests: 3



\## Security behavior

\- Unknown prompt token: HTTP 400

\- Mixed valid and unknown tokens: HTTP 400

\- Empty prompt: HTTP 400

\- Known valid prompts: deterministic RNN template rendering



\## Model

\- Architecture: single\_layer\_leaky\_relu\_rnn

\- Vocabulary size: 28

\- Hidden width: 48

\- Active model: models/multi\_v4\_email\_first/

\- Verified backup: models/multi\_v4\_1\_hardened\_backup\_2026-08-23\_2035/



\## SHA-256 verified artifacts

\- rnn\_meta.json

\- rnn\_Wh.bin

\- rnn\_Wo.bin

\- rnn\_Wx.bin



\## Frozen artifacts

\- server\_rnn.v4\_1\_hardened.ts

\- test/api.chat.test.v4\_1\_hardened.ts

\- test/fixtures/api-cases.v4\_1\_hardened.ts



\## Regression command

npm run test:api

