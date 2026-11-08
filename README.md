# GroundedCheck

GroundedCheck breaks a Vietnamese answer into factual claims and asks an LLM
to check each claim against a supplied source. The JSON report includes the
claim, suggested label, paragraph ID, and verbatim evidence when applicable.

| Label | Meaning |
| --- | --- |
| supported | A source paragraph supports the claim. |
| contradicted | A source paragraph states something incompatible with the claim. |
| insufficient_evidence | No single supplied paragraph establishes support or contradiction. |

Python verifies that each quote appears in its cited paragraph. Reviewers still
need to decide whether the passage supports the label and whether the audit
missed or altered any claims.

## Offline demo

Python 3.11 is sufficient. No extra dependencies or API key are needed.

```powershell
python groundedcheck.py demo
Get-Content reports/demo.json -Encoding utf8
```

Demo mode replays three prepared verdicts against a synthetic store policy.
The expected results are one supported claim, one contradiction, and one claim
with insufficient evidence. The demo checks the report path and citation
validation with fixed data.

## Audit an answer with Groq

Use public or synthetic data. Live mode sends the input text to Groq and can
incur API usage. Local reports include the source text and claims.

```powershell
$secret = Read-Host "Groq API key" -AsSecureString
$env:GROQ_API_KEY = [System.Net.NetworkCredential]::new('', $secret).Password
python groundedcheck.py audit
python groundedcheck.py evaluate
Remove-Item Env:GROQ_API_KEY
```

Audit makes one request for claim extraction and classification. Evaluate makes
nine requests, one per prewritten claim, with gold labels withheld from the model.
Use `--model` if your account uses a different model that supports JSON mode.

```powershell
python groundedcheck.py audit --source data/source.txt --answer data/answer.txt --output reports/my-audit.json
```

UTF-8 TXT files are split at blank lines. IDs such as S2 denote paragraphs,
not page numbers. The prototype sends the full source and limits it to 12,000
characters. Answers are limited to 4,000 characters. The rubric uses one
paragraph per verdict. Cases requiring multiple passages need an extended rubric.

## Evaluate and review

In one live pilot run with `openai/gpt-oss-120b`, the predicted labels matched
the reference labels on all 9 synthetic claims (3 per class) from the bundled
store policy. The report flagged 0 invalid citations. This result describes
one model run on one synthetic document.

The report has a confusion matrix, precision, recall, and support per label.
Undefined precision or recall is represented by null. Invalid quotations are
counted separately from label errors. `source_match` checks the paragraph ID
and quote presence. It does not measure whether the quote proves the claim.

The nine synthetic claims are a small classification pilot, not an evaluation
of claim extraction or general accuracy. For an end-to-end review, compare the
original answer with the extracted claims and record missed or altered claims.
Then inspect the quotes, correct labels in a separate human annotation file,
and keep the original model report for comparison. Include new documents and
claim types before drawing conclusions beyond the sample.

Suggested portfolio experiment: audit an answer from VietRAG Bench, introduce a
wrong number and an unsupported detail, and inspect which errors are detected.
