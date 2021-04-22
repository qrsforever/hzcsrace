
## Errors

### `error: ‘TORCH_CHECK’ was not declared in this scope`

```
sed -i s/"TORCH_CHECK"/"TORCH_CHECK"/g `grep "TORCH_CHECK" -rl .`
```
