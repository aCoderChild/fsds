# TOFU baseline diagnostic verification

Overall: **PASS**

## llama_3_2_3b: PASS


| Split | Questions | Profile blocks | Mean answer probability | Profile-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| forget10 | 400 | 20 | 0.944924 | [0.935612, 0.952716] |
| retain90 | 400 | 20 | 0.945376 | [0.937738, 0.951938] |
| holdout10 | 400 | 20 | 0.146130 | [0.128021, 0.164620] |
| retain90, conflicting question rows removed | 398 | 20 | 0.945705 | [0.938024, 0.952407] |

## llama_3_1_8b: PASS


| Split | Questions | Profile blocks | Mean answer probability | Profile-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| forget10 | 400 | 20 | 0.991221 | [0.989266, 0.992918] |
| retain90 | 400 | 20 | 0.990431 | [0.988918, 0.991831] |
| holdout10 | 400 | 20 | 0.122644 | [0.106545, 0.139419] |
| retain90, conflicting question rows removed | 398 | 20 | 0.990935 | [0.989424, 0.992319] |
