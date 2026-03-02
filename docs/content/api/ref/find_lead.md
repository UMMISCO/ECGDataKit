---
title: "find_lead"
weight: 30
---

`ecgdatakit.processing.leads.find_lead`

**Module:** Lead Derivation

## Signature

```python
find_lead(leads, label)
```

## Description

Find a lead by label (case-insensitive).

## Parameters

<table>
  <thead><tr><th>Name</th><th>Type</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><code>leads</code></td><td><code>list[Lead]</code></td><td>List of leads to search.</td></tr>
    <tr><td><code>label</code></td><td><code>str</code></td><td>Lead label to find (e.g., &quot;II&quot;, &quot;avl&quot;, &quot;V1&quot;).</td></tr>
  </tbody>
</table>

## Example

```python
from ecgdatakit.processing import find_lead

result = find_lead(...)
```
