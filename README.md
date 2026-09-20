# Onramp

A serverless retrieval assistant over Adobe Real-Time CDP documentation, built as a partner-onboarding and solutions-consulting tool. A new SC joining an Adobe practice has to answer prospect questions — *"how do you know a phone visitor and a laptop buyer are the same person?"* — from 1,042 pages of product documentation they have not read yet. Onramp indexes that corpus, retrieves the passages that answer a question, and cites the live Experience League URL for every one, so the answer can be checked rather than trusted. It runs entirely on AWS serverless primitives, deploys with one `cdk deploy`, and every number in this document was measured on the deployed system rather than estimated.

---

## Architecture

```mermaid
%% Fill me in
```

---

## Quickstart

```bash
# 1. toolchain
python3.12 -m venv .venv && ./.venv/bin/pip install -r requirements-dev.txt

# 2. corpus -> S3 (the bucket name comes from the stack output)
export BUCKET=$(aws cloudformation describe-stacks --stack-name OnrampStack \
  --query "Stacks[0].Outputs[?OutputKey=='CorpusBucketName'].OutputValue" --output text)
make load

# 3. deploy everything
make deploy

# 4. ask it something
export API=$(aws cloudformation describe-stacks --stack-name OnrampStack \
  --query "Stacks[0].Outputs[?OutputKey=='ApiEndpoint'].OutputValue" --output text)
export KEY=$(aws apigateway get-api-key --api-key <id> --include-value --query value --output text)
curl -s -X POST "$API" -H "x-api-key: $KEY" -H "Content-Type: application/json" \
  -d '{"question":"How does Real-Time CDP stitch anonymous and authenticated identities?"}'
```

The stack output `GetApiKeyCommand` gives the exact command to retrieve the key. The key value is
never written to a CloudFormation output, because those are readable by anyone with
`cloudformation:DescribeStacks`.

---

## Measured latency

Four conditions, 20 queries each (3 for cold start), measured end to end by `scripts/benchmark.py`.
Wall clock from the caller, not the handler's self-reported time.

| Condition | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| Container cold start | 3 | **12,535 ms** | 12,797 ms | 12,491 ms | 12,797 ms |
| Warm + vector cache | 20 | **233 ms** | 294 ms | 200 ms | 294 ms |
| Warm, cache disabled | 20 | **1,438 ms** | 1,729 ms | 1,410 ms | 1,729 ms |
| Local script path | 20 | **7 ms** | 13 ms | 6 ms | 13 ms |

Four readings worth stating plainly:

**Cold start is 12.5 seconds and it is not hidden.** That is the price of a 1.87 GB container image:
pull, decompress, Python import, then model load. The model itself is only ~1.7 s of it (measured by
running the image with `--network none`), so the bulk is image initialisation. A zip-packaged Lambda
would start in under a second and could not carry the embedding model at all.

**The vector cache is worth 6.2×.** 233 ms against 1,438 ms. That 1.2-second delta is a DynamoDB
`Scan` of 6,112 items, paid on every single call without module-scope caching. This is the single
best-justified design decision in the project, because it is a measurement rather than an argument.

**The local path is 33× faster than the warm API** (7 ms vs 233 ms) running *identical retrieval
code against the identical index*. The 226 ms difference is entirely API Gateway, Lambda invocation,
network, and JSON serialisation. Worth knowing before assuming serverless overhead is negligible on
a workload whose real compute is 7 ms.

**Cold start is stable, not a tail event.** 306 ms of spread across three samples means it is a
reproducible cost you can plan around.

### What provisioned concurrency would cost

Provisioned concurrency keeps execution environments initialised, eliminating the 12.5 s cold start.
Pricing for arm64 in us-east-1:

```
provisioned concurrency : $0.0000041667 per GB-second
memory                  : 2048 MB = 2 GB
seconds per month       : 30 × 24 × 3600 = 2,592,000

1 provisioned instance  = 2 GB × 2,592,000 s × $0.0000041667
                        = $21.60 / month
```

So **$21.60/month per always-warm instance**, before any invocation or duration charges. For a demo
tool answering a handful of questions during a call, that is roughly 100× the cost of the actual
usage, to remove a delay that only the first caller after an idle period experiences. Not worth it
here. It becomes worth it the moment a human is waiting on it live in front of a prospect, which is
exactly the scenario this tool is built for — so the honest answer is "not yet, and here is the
number at which I would."

---

## Retrieval quality

Measured by `scripts/evaluate.py` against `evals/golden.json`: 20 questions researched against the
real Adobe docs, each with an expected source document, plus 5 out-of-scope probes.

| Metric | Local path | Deployed API |
|---|---|---|
| recall@1 | 20% (4/20) | 20% (4/20) |
| recall@3 | 35% (7/20) | 35% (7/20) |
| Refusal rate (out-of-scope) | 40% (2/5) | 40% (2/5) |

**Local and deployed agree exactly**, including the identical list of missed questions. That is the
point of `scripts/local_ask.py` importing `onramp.query` rather than reimplementing it: the laptop
loop measures the deployed system, not a sibling of it.

### 35% is a real number and it deserves an explanation

The eval questions are written the way a prospect actually speaks:

> *"Can the website change what it shows during the same visit, or does the person have to come back
> tomorrow for the personalization to kick in?"*

The document that answers it is titled **Edge segmentation**. There is no shared vocabulary. This is
the hardest case for a bi-encoder, and the eval set is better for containing it.

**Fourteen configurations were measured** before settling (`scripts/ablate_retrieval.py`):

| Model | Chunk | recall@1 | recall@3 | recall@5 |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 500 | 5% | 20% | 30% |
| all-MiniLM-L6-v2 | 220 | 10% | 30% | 30% |
| multi-qa-MiniLM-L6 | 500 | 15% | 25% | 25% |
| multi-qa-MiniLM-L6 | 220 | 15% | 25% | 30% |
| **bge-small-en-v1.5** | **500** | **20%** | **35%** | **35%** |
| bge-small-en-v1.5 | 220 | 10% | 20% | 25% |
| e5-small-v2 | 500 | 10% | 20% | 30% |
| e5-small-v2 | 220 | 15% | 15% | 20% |

Two things that measurement taught, both of which fail *silently*:

1. **Chunk size must match the model's context window.** `all-MiniLM-L6-v2` truncates at 256 word
   pieces. Chunking at ~500 tokens meant **half of every chunk was never embedded** — no error, no
   warning, just quietly worse retrieval. That is why shrinking chunks helped MiniLM and hurt bge,
   whose window is 512. `max_seq_tokens` is now a property of the provider, not a free config value.
2. **Retrieval models need their instruction prefix.** bge wants one on the query side and must not
   have it on passages. Hence `embed_query()` alongside `embed()`.

### Two hypotheses that were wrong

Reported because a negative result that cost real effort is worth more than an unexamined number.

**Cross-encoder reranking did not help.** `ms-marco-MiniLM-L-6-v2` over the top-N made results worse
at every depth beyond 20 (`scripts/ablate_rerank.py`). The measurement is also confounded: the
cross-encoder's 512-token window covers the question *and* passage combined, so 500-token passages
were being truncated — the same class of bug as (1) above. The honest status is "not demonstrated to
help, and not cleanly measured."

**Hybrid BM25 + dense did not help either.** Every RRF weighting scored below dense alone
(`scripts/ablate_hybrid.py`), and BM25 alone managed 15% recall@3. The reasoning was that lexical
matching would rescue the deep misses via distinctive terms like "Destination SDK". It did not,
because **the questions deliberately avoid Adobe's vocabulary** — there is no rare term in "can the
website change what it shows during the same visit" for BM25 to match. BM25 helps when the *user*
supplies the rare term; here the rare terms exist only in the documents. The implementation is kept
(`src/onramp/lexical.py`, 17,817 terms, 0.3 s build, 5 ms/query) as a documented negative result.

**What would likely work next** is query expansion / HyDE: generate a hypothetical answer with an
LLM and embed *that* instead of the question, which attacks the vocabulary gap directly. It requires
generation, which this account's Bedrock quota defect blocks entirely. See *Provider portability*.

### The refusal threshold is calibrated, and it has a ceiling

Cosine scores for out-of-scope probes against in-scope questions:

```
out-of-scope   0.5594 – 0.7277
in-scope min   0.6263
```

**The distributions overlap.** The competitive probe — *"We're also looking at Salesforce Data Cloud.
Which one resolves identity better?"* — scores **0.7277**, higher than 8 of the 20 legitimate
questions. It should: it genuinely is a question about identity resolution, and the index genuinely
contains identity documents. Cosine similarity measures topical relatedness, not answerability.

| Threshold | Refuses out-of-scope | Wrongly refuses in-scope |
|---|---|---|
| 0.30 (naive default) | 0/5 | 0/20 |
| **0.62 (chosen)** | **2/5** | **0/20** |
| 0.65 | 2/5 | 6/20 |
| 0.70 | 4/5 | 13/20 |
| 0.73 | 5/5 | 16/20 |

0.62 is the last point where refusals are free. Catching the remaining three — pricing, competitive,
customer-specific — needs a classifier, not a cutoff. **40% is the honest ceiling for a similarity
threshold on this workload.**

---

## Citation URL accuracy

**Pass rate: 100% (20/20).** `scripts/verify_citations.py` samples stored `doc_url` values from
DynamoDB — not recomputed from the mapper, which would test the function against itself — and issues
HTTP HEAD requests against the live site.

Getting there found three real defects, and the mapping rule in the original design was one of them.

### The obvious rule is wrong

```
help/<path>.md  ->  .../experience-platform/<path>
```

Checked against the 20 hand-researched URLs: **13/20 correct, 7 wrong.** Experience League publishes
by `TOC.md` structure, not filesystem path:

```
# Adobe Experience Platform Identity Service {#identity}
- Features {#features}
  - [Identity namespace](./features/namespaces.md)
```

The guide heading's `{#anchor}` becomes the first URL segment and each nested section contributes its
own. One rule fixes all seven:

| Case | Filesystem | Published |
|---|---|---|
| Guide anchor | `identity-service/guardrails.md` | `/identity/guardrails` |
| Section insert | `identity-service/identity-graph-linking-rules/overview.md` | `/identity/**features**/identity-graph-linking-rules/overview` |
| Anchor overrides path | `sources/tutorials/ui/create/streaming/http.md` | `/sources/**ui-tutorials**/create/streaming/http` |

### Three defects the verification caught

| Defect | Scope | Detection |
|---|---|---|
| `destinations/TOC.md` never synced to S3 | **287 docs** silently used the naive rule | 404 in sample |
| rtcdp TOC links absolutely (`/help/rtcdp/…`) | never matched the lookup key | 404 in sample |
| **60 documents are unpublished upstream** | `hide: true` / `hidefromtoc: yes` | 404 with no valid URL |

The third is the interesting one. Those 60 documents exist in the git repo but Adobe does not serve
them — there is no correct URL, so every citation to one was a guaranteed 404. They are now filtered
at ingest, which is why the corpus is 1,042 documents rather than 1,102.

Every one of these was silent. Nothing errored; the URLs were simply wrong. That is the argument for
the verification script existing at all.

---

## Data model

One DynamoDB table, on-demand billing, PITR enabled.

```
PK = DOC#<sha256 of s3 key>      SK = CHUNK#<zero-padded index>
```

Hashing the key keeps the partition key a fixed 64 characters regardless of source path depth.
Zero-padding means `CHUNK#000009` sorts before `CHUNK#000010`.

Attributes: `text`, `embedding`, `title`, `product_area`, `heading_path`, `s3_key`, `doc_url`,
`token_count`, `embedding_provider`, `embedding_dimensions`.

### Why the embedding is Binary, not a list of Numbers

DynamoDB serialises each Number as a decimal string — roughly 20 bytes per float once the type
wrapper is counted. Packed little-endian float32 is 4 bytes.

| | 384 dims (bge-small) | 1024 dims (Titan V2) |
|---|---|---|
| List of Numbers | ~7.7 KB | ~20.5 KB |
| **Packed float32** | **1,536 B** | **4,096 B** |
| Saving | 5× | 5× |

Both are far under the **400 KB item limit** — 0.4% and 1.0% respectively — so the limit is not the
constraint. Scan throughput is: the query path reads every item on a cold load, and 5× less bytes is
5× less to transfer. Measured mean item size is **4,491 bytes** (the chunk text dominates, not the
vector), giving a **~26 MB table** over 6,112 items.

Switching to Titan's 1024 dims would add ~2.5 KB per item, roughly **+15 MB** — still trivial in
storage terms, but it triples the vector payload the query path deserialises on every cold load.
That cost is documented here *before* the swap rather than discovered after it.

---

## Cost model

us-east-1, arm64. Query path only; ingest is one-off.

Rates below were **fetched from the AWS Price List API**, not recalled:
`Lambda ARM Tier-1 $0.0000133334/GB-s`, `API Gateway REST $3.50/M (first 333M)`,
`ECR storage $0.10/GB-month`. Re-check with `aws pricing get-products` before quoting them
elsewhere; the arithmetic is shown so a stale rate is a one-line correction.

**Per-query resource use, measured:** 233 ms warm at 2048 MB.

```
Lambda duration : 0.233 s × 2 GB × $0.0000133334/GB-s  = $0.00000621
Lambda request  : $0.20 per 1M                          = $0.00000020
API Gateway REST: $3.50 per 1M                          = $0.00000350
DynamoDB reads  : cached in module scope; ~0 per warm call
                                                   total ≈ $0.0000099 / query
```

| | 1,000 queries/month | 100,000 queries/month |
|---|---|---|
| Lambda duration | $0.0062 | $0.62 |
| Lambda requests | $0.0002 | $0.02 |
| API Gateway (REST) | $0.0035 | $0.35 |
| DynamoDB on-demand reads | ~$0.00 | ~$0.02 |
| **Query subtotal** | **$0.01** | **$1.01** |
| DynamoDB storage (26 MB × $0.25/GB) | $0.0065 | $0.0065 |
| S3 storage (12 MB × $0.023/GB) | $0.0003 | $0.0003 |
| **ECR storage (3 images × 1.5 GB × $0.10/GB)** | **$0.45** | **$0.45** |
| CloudWatch Logs (7-day retention) | ~$0.05 | ~$0.50 |
| **Total** | **≈ $0.52 / month** | **≈ $2.47 / month** |

**ECR storage dominates the small case — 87% of the bill at 1k queries.** Three 1.5 GB compressed
images cost more than a thousand queries. That is the container-image tradeoff showing up on the
invoice, and it is why the lifecycle policy matters: before it was applied, **8 images had
accumulated to 4.50 GB ($0.45/month growing unbounded)**. Every `cdk deploy` that touches
`src/onramp/` pushes a new one.

Both scenarios are under $3/month. The design decision that would matter at scale is the DynamoDB
full-scan retrieval, not the per-query cost.

---

## Design tradeoffs

### The EmbeddingProvider abstraction, and why it exists

**I hit a Bedrock quota provisioning defect and chose portability over waiting.**

Every on-demand Bedrock inference quota on this account reads `0.0` and is marked non-adjustable —
82 of 84 models, including every Claude and Titan model. `InvokeModel` returns `ThrottlingException`
despite `authorizationStatus: AUTHORIZED` and `entitlementAvailability: AVAILABLE`. Service Quotas
rejects increase requests outright: *"The request failed because the specified quota is not
adjustable."* A support case is filed.

The alternative to waiting was an abstraction: `EmbeddingProvider` is a `Protocol` with three
implementations, selected by one CDK context value and one env var. The system ships on a local
model, and `BedrockEmbeddingProvider` is written and unit-tested against a mocked boto3 client so
the swap is a config change rather than a rewrite. **Nothing in the default deploy can call
Bedrock** — the IAM grant is gated behind the same context flag, so the quota defect cannot affect
the deployed system even by accident.

A constraint became an architecture decision. That is a better outcome than a blocked project.

### Local MiniLM/bge 384 dims vs Bedrock Titan 1024 dims

| | bge-small-en-v1.5 (shipped) | Titan Text Embeddings V2 |
|---|---|---|
| Dimensions | 384 | 1024 |
| Quality | 35% recall@3 measured | unmeasurable — quota is 0 |
| Storage per vector | 1,536 B | 4,096 B (2.7×) |
| Table size | ~26 MB | ~41 MB |
| Cost per embedding | $0 (CPU in-process) | $0.00002 / 1k tokens |
| Batch API | one forward pass for a list | **one HTTP call per text** |
| Cold start | +1.7 s model load | none (no model in image) |
| Image size | 1.87 GB | would drop to ~300 MB |
| Quota | none | blocked on this account |

The honest comparison is incomplete: Titan's retrieval quality is **unmeasured** because it cannot be
invoked. Titan would shrink the image dramatically and eliminate the model-load cold start, at the
cost of a network round trip per embedding and, during ingest, one HTTP call per chunk rather than
one batched forward pass — 6,112 sequential calls instead of ~1,100 batches.

### Container image Lambda vs zip Lambda

Zip Lambdas are capped at 250 MB unzipped. torch alone exceeds that, so a local embedding model is
not possible in a zip — the choice was forced by the provider decision. **Cost: 12.5 s cold start vs
well under 1 s.** Benefit: any dependency, no layer juggling, and the exact same image runs locally
for testing (verified with `docker run --network none`). If Bedrock quota arrived, the model leaves
the image, the image drops to ~300 MB, and zip packaging becomes viable again.

### DynamoDB brute-force cosine vs OpenSearch Serverless vs S3 Vectors

Currently 6,112 vectors × 384 dims = ~9 MB of float32 in memory. Scoring is a single numpy matmul,
sub-millisecond. Loading them is 1.2 s (measured), amortised to zero by the module-scope cache.

**The decision flips at roughly 50,000–100,000 chunks**, where the cold-load scan exceeds a few
seconds and the matrix stops comfortably fitting in Lambda memory alongside the model. That is
~10–20× the current corpus, or all of Experience League rather than nine guides.

OpenSearch Serverless is the obvious next step, and its cost model is the reason not to take it yet.
It bills per OCU-hour with a **minimum number of always-on OCUs**, so it has an idle floor this
system does not:

```
published rate (NOT verified against the Price List API -- it is not exposed
there; confirm on the OpenSearch Serverless pricing page before quoting)

  $0.24 per OCU-hour × 730 hours          = $175 / OCU-month
  4 OCUs (2 indexing + 2 search, the
  documented production minimum)          ≈ $700 / month
  2 OCUs (dev/test minimum)               ≈ $350 / month
```

Either figure is two to three orders of magnitude above this system's **$0.52–$2.47/month**, and it
is charged whether anyone asks a question or not. That is what makes it indefensible at 6,112
vectors, not the technology.

S3 Vectors is the more interesting option: purpose-built for this, pay-per-use, no idle floor. The
reason to stay on DynamoDB today is that it is already the metadata store, so retrieval needs no
second system to keep consistent.

### SQS between S3 and Lambda, rather than S3 invoking Lambda directly

Direct S3→Lambda has no batching, no retry policy you control, and no dead-letter queue. A bulk sync
of 1,111 objects would become 1,111 near-simultaneous invocations. With SQS the same sync becomes a
drainable backlog processed at batch size 5, with `maxReceiveCount: 3` and a DLQ.

Measured: the real drain took **540 s for 1,111 messages, with 0 DLQ entries** across four full
re-ingests. Partial batch responses mean one malformed document fails alone rather than forcing
redelivery of the four that succeeded alongside it.

### Retrieval mode vs generated mode

**Retrieval-only ships; generation is behind a flag that is off.** For a pre-sales tool this is
defensible on its own merits, not merely as a workaround:

- Every word shown is Adobe's, with a URL. There is no surface for a hallucinated product claim.
- An SC repeating a generated answer to a prospect owns that answer. An SC reading a cited passage
  can point at the source.
- Generation adds seconds to a 233 ms path.

The `ChatProvider` Protocol and two stubs exist so the capability is a config change. Both raise
`NotImplementedError` rather than silently degrading, and no deploy depends on either.

### Fixed-size vs heading-aware chunking

This corpus has deep hierarchies and heavy code/table content. Fixed-size chunking splits fenced code
blocks and separates markdown tables from their header rows — half a code sample retrieves as
confidently as a whole one and is worse than useless. The chunker treats fences and tables as atomic
and carries `product_area | title | heading_path` into the embedded text, so a chunk reading "Select
**Create audience**" still embeds near "segmentation".

Oversized atomic blocks become their own chunk rather than being split: one large chunk beats two
broken ones.

### REST API with usage plans rather than HTTP API

HTTP APIs cost **$1.00/M** against REST's **$3.50/M** — 3.5× cheaper — and have lower latency. They
have no native API keys or usage plans. Throttling a demo tool at 5 rps / 10 burst and issuing a key
is the requirement, and rebuilding that with a Lambda authorizer plus a rate-limit store would cost
more in complexity than $2.50/M saves at this volume. At 100k queries/month the premium is **$0.25**.

### IAM Identity Center short-lived credentials vs static access keys

Everything here runs under `AWSReservedSSO_AdministratorAccess` via IAM Identity Center. The
credentials expire and cannot be committed, because they do not exist as a file to commit. Static
access keys are the single most common source of leaked AWS credentials on GitHub.

The operational cost is real — the session expires mid-work and needs `aws sso login` — and it is
worth paying. Lambda never uses either: the functions assume execution roles scoped by CDK's
`grant_*` helpers to specific bucket and table ARNs.

---

## Provider portability

What a Bedrock swap changes, and what it does not:

| Changes | How |
|---|---|
| Provider selection | `cdk deploy -c embeddingProvider=bedrock` |
| Env var | `EMBEDDING_PROVIDER=bedrock` (set by CDK from the context value) |
| IAM | `bedrock:InvokeModel` granted, scoped to foundation-model and inference-profile ARNs — gated behind the same flag |
| Dimensions | 384 → 1024, recorded per item as `embedding_dimensions` |
| Storage | 1,536 B → 4,096 B per vector; table ~26 MB → ~41 MB |
| Re-ingest | **Required.** Mixed-provider tables are refused by the query path |
| Image | Model leaves the image; ~1.87 GB → ~300 MB; cold start drops |

| Does NOT change | Why |
|---|---|
| Chunker | Operates on markdown, not on vectors |
| Citation mapping | Derived from TOC structure, independent of embeddings |
| Retrieval logic | Cosine over a matrix, whatever produced it |
| Eval set | Questions and expected URLs are model-agnostic |
| API contract | Same response shape; only `provider` changes value |
| DynamoDB schema | Same attributes; `embedding` is bytes either way |

### The mixed-provider guard

Every chunk records `embedding_provider` and `embedding_dimensions`. The query path compares before
scoring and raises `ProviderMismatchError` rather than returning results.

The dangerous case is **not** a dimension mismatch — numpy raises on that, loudly. It is a
*same-dimension, different-model* index, or a table half-migrated after a provider switch. Cosine
similarity between a MiniLM vector and a bge vector is arithmetic that succeeds and means nothing:
noise wearing the costume of a score. A pre-sales tool citing the wrong Adobe doc with a confident
0.76 is worse than one that refuses to start.

This is not hypothetical — it happened during development. Migrating MiniLM → bge left the table
mixed for ~4 minutes mid-drain, and both models are 384 dims, so only the provider string
distinguished them.

---

## Service mapping

| AWS | Google Cloud | Azure |
|---|---|---|
| S3 | Cloud Storage | Blob Storage |
| SQS | Pub/Sub | Service Bus Queues |
| SQS DLQ | Pub/Sub dead-letter topic | Service Bus dead-letter queue |
| Lambda (container) | Cloud Run / Cloud Run functions | Container Apps / Functions (container) |
| DynamoDB | Firestore / Bigtable | Cosmos DB |
| API Gateway (REST) | API Gateway / Apigee | API Management |
| ECR | Artifact Registry | Container Registry |
| CloudFormation / CDK | Deployment Manager / Terraform | ARM / Bicep |
| IAM Identity Center | Cloud Identity + IAM | Entra ID |
| X-Ray | Cloud Trace | Application Insights |
| CloudWatch Logs | Cloud Logging | Monitor Logs |
| Bedrock | Vertex AI | Azure AI Foundry |
| S3 event notifications | Eventarc / GCS notifications | Event Grid |

---

## Teardown

```bash
make destroy          # cdk destroy; empties and removes the corpus bucket
```

The stack sets `RemovalPolicy.DESTROY` with `auto_delete_objects` on the bucket and `DESTROY` on the
table and log groups, so nothing billable survives. This is safe *because* the corpus is reproducible
from a public git repo; it would be wrong for a bucket holding originals.

Two things `cdk destroy` does not remove:

- **ECR images** in the shared CDK bootstrap repo. The lifecycle policy caps them at 3, but removing
  the last of them means `cdk bootstrap --force` or deleting the repo by hand.
- **The CDKToolkit bootstrap stack**, shared by every CDK app in the account.

```bash
aws ecr list-images --repository-name cdk-hnb659fds-container-assets-<account>-<region>
```

---

## Attribution and license

Adobe Experience Platform documentation is sourced from
[AdobeDocs/experience-platform.en](https://github.com/AdobeDocs/experience-platform.en), which is
**MIT licensed** (© Copyright 2021 Adobe). The corpus is not redistributed in this repository — it is
fetched at build time and `.gitignore` excludes any cloned Adobe content. Citations link to the live
Experience League pages rather than reproducing them.

This project's own code is MIT licensed.

**Disclaimer:** Onramp is an independent personal project built for learning. It is **not affiliated
with, endorsed by, or sponsored by Adobe**. "Adobe", "Experience Platform", and "Real-Time CDP" are
trademarks of Adobe Inc. Nothing here is an official Adobe tool or an authoritative source on Adobe
products; the cited documentation is.
