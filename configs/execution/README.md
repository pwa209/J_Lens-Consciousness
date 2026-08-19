# Execution participant table

`participants.tsv` remains header-only until the AutoDL one-participant
inspection for each dataset has succeeded. Add one row per source participant:

```text
dataset_id	participant_id	include	reason
kronemer	001	1	eligible source recording
```

Do not infer identifiers from archive filenames without comparing them with the
source event and participant metadata.

