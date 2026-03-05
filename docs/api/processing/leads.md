# Lead Derivation

Pure numpy — no scipy required. Validates sample rate and length match.

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
