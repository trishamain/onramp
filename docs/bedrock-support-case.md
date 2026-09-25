# AWS Support case: Bedrock on-demand inference quota

Ready to paste into Support Center. Evidence captured from the account on 2026-09-25 03:13 UTC.

| Field | Value |
|---|---|
| Type | Service limit increase (see note below if only Account cases are available) |
| Service | Amazon Bedrock |
| Region | us-east-1 |
| Account | 048834207434 |
| Severity | General guidance |

**Subject:** On-demand inference quotas are 0 and non-adjustable across all Bedrock models in us-east-1

---

## Description

Account 048834207434 has no on-demand inference capacity for Amazon Bedrock in us-east-1. Every
on-demand quota reads 0.0 and is marked non-adjustable, so InvokeModel and Converse fail with
ThrottlingException, and Service Quotas rejects increase requests outright. There is no
self-service path to resolve this.

Batch inference quotas on the same account are provisioned normally, and the models are fully
entitled, so this is specifically a missing on-demand allocation rather than a dormant or
unverified account.

Please provision standard on-demand inference quotas for this account in us-east-1.

### Evidence

```
BEDROCK QUOTA EVIDENCE - account 048834207434, us-east-1
captured 2026-09-25T03:12:50Z

1. ON-DEMAND QUOTAS ARE ZERO AND NON-ADJUSTABLE
   requests/min quotas : 84 total, 82 are 0.0, 2 non-zero
   tokens/min quotas   : 63 total, 61 are 0.0
   adjustable          : 2 of 84
   the only non-zero on-demand allocations in the entire account:
          1.0  On-demand model inference requests per minute for AI21 Labs Jamba 1.5 Large
          1.0  On-demand model inference requests per minute for AI21 Labs Jamba 1.5 Mini

   the two this project needs:
          0.0  adjustable=False  L-26C560CE  On-demand model inference requests per minute for Amazon Titan Text Embeddings V2
          0.0  adjustable=False  L-DE641971  On-demand model inference tokens per minute for Amazon Titan Text Embeddings V2

2. BATCH QUOTAS ARE POPULATED - the account is otherwise provisioned
   60 batch job-size quotas at 5.0 GB, e.g.:
          5.0  Batch inference job size (in GB) for Gemma 3 12B
          5.0  Batch inference job size (in GB) for Claude Sonnet 4.5
          5.0  Batch inference job size (in GB) for OpenAI GPT OSS 120b
   So this is specifically an on-demand inference allocation gap,
   not a dormant or unprovisioned account.

3. NO INCREASE HAS EVER BEEN REQUESTED ON THIS ACCOUNT
   prior requests: 0

4. THE MODEL IS FULLY ENTITLED - authorized but unusable
   authorizationStatus entitlement region agreement:  AUTHORIZED  AVAILABLE  AVAILABLE  AVAILABLE

5. SERVICE QUOTAS REFUSES THE SELF-SERVICE PATH
   aws: [ERROR]: An error occurred (IllegalArgumentException) when calling the RequestServiceQuotaIncrease operation: The request failed because the specified quota is not adjustable.

6. LIVE INVOCATION FAILS
   RequestId fa80ba76-815d-4517-a66f-201e24067423
   ThrottlingException: Too many requests (reached max retries: 2)
   full redacted wire trace: bedrock-throttle-trace.redacted.txt
```

### What this rules out

- **Not a model-access problem.** `get-foundation-model-availability` returns
  `AUTHORIZED / AVAILABLE / AVAILABLE / AVAILABLE` for the model being called.
- **Not the Anthropic use-case form.** It was submitted and accepted; the earlier
  `ResourceNotFoundException: Model use case details have not been submitted` no longer occurs,
  and `get-use-case-for-model-access` returns the stored submission.
- **Not IAM.** The caller is `AdministratorAccess` via IAM Identity Center;
  `sts get-caller-identity` succeeds and every other API on the account works.
- **Not an unrequested increase.** `list-requested-service-quota-change-history` returns 0 prior
  requests for the bedrock service code, because `RequestServiceQuotaIncrease` rejects these
  quotas as non-adjustable (item 5 above). The empty request history is a consequence of that
  refusal, not a reason for the zero allocation. There is no request for me to submit.
- **Not a dormant account.** 60 batch inference job-size quotas are provisioned at 5.0 GB.
- **Not transient throttling.** Reproduced across four days with no successful Bedrock
  invocation ever recorded on the account.

### Request

Please provision standard on-demand inference quotas in us-east-1, in particular:

- `amazon.titan-embed-text-v2:0` — quota codes `L-26C560CE` (requests/min) and `L-DE641971`
  (tokens/min)
- Anthropic Claude models accessed through `us.*` cross-region inference profiles

If this is expected behaviour pending a further activation step, please confirm what that step is
and the expected timeline.

Please note that asking me to submit a Service Quotas increase request is not an available
resolution: the quotas in question are flagged non-adjustable and the API refuses the call, as
shown in item 5. Seeding these quotas is a service-side operation.

### Attachments

- `bedrock-quota-evidence.txt` — the quota state above
- `bedrock-throttle-trace.redacted.txt` — full wire trace of the failing call, RequestId
  `fa80ba76-815d-4517-a66f-201e24067423`. SigV4 credentials and session tokens are redacted.

### Note on case type

Basic support cannot open Technical cases. If "Service limit increase" is unavailable, file this as
an **Account** case — new-account capacity provisioning legitimately qualifies, and the evidence
above is what matters, not the category.

---

## Why this matters to this repository

Onramp was designed around this defect rather than blocked by it. The `EmbeddingProvider`
abstraction in `src/onramp/providers/` exists because on-demand Bedrock was unusable: the system
ships on a local model, `BedrockEmbeddingProvider` is written and unit-tested against a mocked
client, and the deployed stack grants no `bedrock:*` IAM at all unless a CDK context flag selects
it. Quota arriving later is a config change plus a re-ingest — see *Provider portability* in the
README.
