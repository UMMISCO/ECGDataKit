# Anonymization

Classes of `ecgdatakit.anonymize`. Usage, options and the catalog are described in {doc}`../anonymization`.


Import: `from ecgdatakit.anonymize import Anonymizer, Report, Catalog, DatasetLocked`

```python
from ecgdatakit.anonymize import Anonymizer

reports = Anonymizer("/path/to/source", datasets=True, patients_dir_name="RAW").run()
for report in reports:
    print(report.dataset, report.anonymized, report.failed, report.errors)
```

```{eval-rst}
.. currentmodule:: ecgdatakit.anonymize

.. autoclass:: Anonymizer
   :members: run, run_dataset, datasets, dataset_of, catalog_path, check

.. autoclass:: Report
   :members:
   :undoc-members:

.. autoclass:: Catalog
   :members: get, put, active, save

.. autoexception:: DatasetLocked
```
