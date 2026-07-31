# security

## reporting a vulnerability

use github private vulnerability reporting — the security tab of this
repository, "report a vulnerability". please don't open a public issue for
something exploitable. only the latest release is supported.

## what the risk surface is

findyourcode is a local cli. by default it talks to no network, and the only
thing it writes is `.findyourcode/index.db`.

- that file holds your code — chunk text and summaries in plaintext, as
  sensitive as the repository itself. keep `.findyourcode/` gitignored.
- `--provider voyage` and `--provider openai` send the text of every indexed
  chunk to a third party over https, with `VOYAGE_API_KEY` / `OPENAI_API_KEY`
  read from the environment and `OPENAI_BASE_URL` deciding where openai
  requests go. a remote provider means your code leaves the machine.
