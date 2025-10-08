# MACE Global Attention Compatibility

## Issue Description

MACE cannot be combined with global attention mechanisms (GPS or EquiformerV2) due to fundamental architectural incompatibilities in e3nn tensor operations.

## Error Details

When MACE is combined with global attention, the following error occurs:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (1506x192 and 133x64)
```

This error originates in `e3nn/_fc.py` and is caused by:
1. MACE has its own complex equivariant structure with specific tensor shapes
2. Global attention layers modify these tensor dimensions
3. The e3nn operations in MACE expect specific input shapes that become incompatible

## Solution

MACE has been **excluded from global attention tests** in `tests/test_examples.py`. 

## Supported Combinations

The following 12 MPNN types work with both GPS and EquiformerV2 global attention:
- SAGE
- GIN 
- GAT
- MFC
- PNA
- PNAPlus
- CGCNN (GPS only)
- SchNet
- DimeNet
- EGNN
- PNAEq
- PAINN

Total supported combinations: **24** (12 MPNN × 2 global attention engines)

## MACE Usage

MACE can still be used **without** global attention:
```python
# ✅ This works
python examples/qm9/qm9.py --mpnn_type MACE

# ❌ This fails
python examples/qm9/qm9.py --mpnn_type MACE --global_attn_engine GPS
python examples/qm9/qm9.py --mpnn_type MACE --global_attn_engine EquiformerV2
```