
```
pip install "ray[default]"
```

```
ray job submit --runtime-env ray_config.yaml --address http://127.0.0.1:8265 --working-dir lm_buddy_config -- lm-buddy run finetuning --config finetuning_config.yaml
```


