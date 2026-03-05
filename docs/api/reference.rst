Function Reference
==================

.. _api-processing-ref:

Processing Functions
--------------------

.. currentmodule:: ecgdatakit.processing

ECG Cleaning
~~~~~~~~~~~~

.. autofunction:: clean_ecg
   :noindex:

Filters
~~~~~~~

.. autofunction:: lowpass
   :noindex:
.. autofunction:: highpass
   :noindex:
.. autofunction:: bandpass
   :noindex:
.. autofunction:: notch
   :noindex:
.. autofunction:: remove_baseline
   :noindex:
.. autofunction:: diagnostic_filter
   :noindex:
.. autofunction:: monitoring_filter
   :noindex:

Resampling
~~~~~~~~~~

.. autofunction:: resample
   :noindex:

Normalization
~~~~~~~~~~~~~

.. autofunction:: normalize_minmax
   :noindex:
.. autofunction:: normalize_zscore
   :noindex:
.. autofunction:: normalize_amplitude
   :noindex:

R-Peak Detection
~~~~~~~~~~~~~~~~

.. autofunction:: detect_r_peaks
   :noindex:
.. autofunction:: heart_rate
   :noindex:
.. autofunction:: rr_intervals
   :noindex:
.. autofunction:: instantaneous_heart_rate
   :noindex:

Heart Rate Variability
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: time_domain
   :noindex:
.. autofunction:: frequency_domain
   :noindex:
.. autofunction:: poincare
   :noindex:

Transforms & Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: power_spectrum
   :noindex:
.. autofunction:: fft
   :noindex:
.. autofunction:: segment_beats
   :noindex:
.. autofunction:: average_beat
   :noindex:

Signal Quality
~~~~~~~~~~~~~~

.. autofunction:: signal_quality_index
   :noindex:
.. autofunction:: classify_quality
   :noindex:
.. autofunction:: snr_estimate
   :noindex:

Lead Derivation
~~~~~~~~~~~~~~~

.. autofunction:: derive_lead_iii
   :noindex:
.. autofunction:: derive_augmented
   :noindex:
.. autofunction:: derive_standard_12
   :noindex:
.. autofunction:: find_lead
   :noindex:


.. _api-plotting-ref:

Plotting Functions
------------------

.. currentmodule:: ecgdatakit.plotting

Static Plots (matplotlib)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: plot_lead
   :noindex:
.. autofunction:: plot_leads
   :noindex:
.. autofunction:: plot_12lead
   :noindex:
.. autofunction:: plot_peaks
   :noindex:
.. autofunction:: plot_beats
   :noindex:
.. autofunction:: plot_average_beat
   :noindex:
.. autofunction:: plot_spectrum
   :noindex:
.. autofunction:: plot_spectrogram
   :noindex:
.. autofunction:: plot_rr_tachogram
   :noindex:
.. autofunction:: plot_poincare
   :noindex:
.. autofunction:: plot_hrv_summary
   :noindex:
.. autofunction:: plot_quality
   :noindex:
.. autofunction:: plot_report
   :noindex:

Interactive Plots (plotly)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: iplot_lead
   :noindex:
.. autofunction:: iplot_leads
   :noindex:
.. autofunction:: iplot_12lead
   :noindex:
.. autofunction:: iplot_peaks
   :noindex:
.. autofunction:: iplot_spectrum
   :noindex:
.. autofunction:: iplot_rr_tachogram
   :noindex:
.. autofunction:: iplot_poincare
   :noindex:
.. autofunction:: iplot_report
   :noindex:
