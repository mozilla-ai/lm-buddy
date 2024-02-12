## Working with lm-buddy

Submitting an `lm-buddy` job includes two parts: 
a YAML file that specifies configuration for your finetuning or evaluation job, 
and a driver script that either invokes the `lm-buddy` CLI directly 
or submits a job to Ray that invokes `lm-buddy` as its entrypoint.

## Examples

For a full end-to-end interactive workflow running from within `lm-buddy`, 
see the sample notebooks under `notebooks`.
