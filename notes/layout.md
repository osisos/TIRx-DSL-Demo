# Layout 的设计

## 典型问题

我们考虑一个典型的在`sm89`硬件上矩阵乘的计算。

首先看最内层，`inner_layout`，因为`sm89`接受的硬件

```python
Tile_
```
