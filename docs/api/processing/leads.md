# Lead Derivation

Pure numpy, no scipy required. Validates that sample rates and lengths match.

- Raw leads with the same `resolution` and `resolution_unit` are combined in ADC counts (offsets folded in), and the derived leads stay raw on that scale.
- Otherwise both leads are converted to physical values in lead I's unit first, and the derived leads are physical. `derive_standard_12` then also converts I, II and V1 to V6, so all 12 leads share one scale.
- Raw leads with different scales and no voltage unit raise `ValueError`.
- Derived leads do not copy file-specific metadata from lead I: `annotations`, `quality`, `transducer` and `adc_resolution` are reset.

| | |
|---|---|
| {func}`~ecgdatakit.processing.derive_lead_iii` | Derive Lead III from Leads I and II (Einthoven's law) |
| {func}`~ecgdatakit.processing.derive_augmented` | Derive augmented limb leads aVR, aVL, aVF |
| {func}`~ecgdatakit.processing.derive_standard_12` | Assemble a full 12-lead ECG |
| {func}`~ecgdatakit.processing.find_lead` | Find a lead by label (case-insensitive) |

```{eval-rst}
.. currentmodule:: ecgdatakit.processing

.. autofunction:: derive_lead_iii
.. autofunction:: derive_augmented
.. autofunction:: derive_standard_12
.. autofunction:: find_lead
```
